"""
signals.py v12 — Institutional-Grade Signal Engine
All v11 scorecard gaps fixed:
  1. RSI        — adjust=False + explicit 100/0 edge case        (7 → 9)
  2. MACD       — adjust=False, single-pass crossover, histogram  (6 → 9)
  3. Bollinger  — bb_pos clamped [0,1], bandwidth + squeeze       (6 → 8)
  4. ATR        — Wilder's EWM smoothing (matches Zerodha/TV)     (6 → 9)
  5. Supertrend — numpy array loop, Wilder ATR, mult 2.5          (5 → 9)
  6. VWAP       — 20-day rolling + price_vs_vwap %                (4 → 8)
  7. EMA/Trend  — slope check, momentum-fading flag, EMA200 back  (7 → 8)
  8. Fibonacci  — swing-peak based (scipy), not fixed window      (6 → 8)
  9. Risk Engine— find_sector_picks + scanner now use unified     (7 → 9)
 10. Liquidity  — soft gate with liquidity_ok flag (no silent None)(7 → 8)
All output keys are backward-compatible with app.py.
"""

import os
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.signal import find_peaks

# ==============================================================================
# 1. MULTI-SOURCE NSE UNIVERSE LOADER
# ==============================================================================
#
# Place CSV files in the SAME directory as signals.py (repo root).
#
# Download links (browser → CSV button):
#  Nifty 500:    nseindia.com/products-services/indices-nifty500-index
#  Nifty 1000:   nseindia.com/products-services/indices-nifty-indices-nifty1000
#  Midcap 150:   nseindia.com/products-services/indices-nifty-midcap-150-index
#  Smallcap 250: nseindia.com/products-services/indices-nifty-smallcap-250-index
#  Microcap 250: nseindia.com/products-services/indices-nifty-microcap-250-index
#  All NSE:      nseindia.com/market-data/live-equity-market → Download (EQ series)
#
# Supported column variants are auto-detected — no manual editing needed.
# ==============================================================================

# ── Absolute base directory — ALWAYS reliable regardless of cwd ──────────────
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── MARKET CONFIG (US standalone build) ──────────────────────────────────────
MARKET     = "US"
SUFFIX     = ""             # US tickers are bare — no .NS/.BO suffix
BENCH_NAME = "S&P 500"      # RS + regime benchmark (display name)
BENCH_SYM  = "^GSPC"        # S&P 500 index symbol on Yahoo
CURRENCY   = "$"

# Live scanners fetch at most this many symbols (most-liquid-first + all ETFs).
# Scanning the full ~4,800-name universe live rate-limits Yahoo on Streamlit
# Cloud, which is why scanners returned nothing. 800 keeps scans fast/reliable.
# Raise it after baking us_universe_liquid.csv (run build_us_universe.py).
MAX_SCAN_SYMBOLS = 800

def _yahoo(sym):
    """Resolve a bare symbol to its Yahoo ticker for the active market.
    For US, share-class dots use Yahoo's dash convention (BRK.B -> BRK-B)."""
    clean = sanitize_ticker(sym)
    if SUFFIX == "":            # US: Yahoo uses '-' for share classes
        clean = clean.replace(".", "-")
    return clean + SUFFIX

# ── CSV configs: (filename, label, series_filter or None) ────────────────────
# Symbol and sector columns are auto-detected from the actual file.
# US universe: prefer the liquid-filtered file (built by build_us_universe.py),
# else fall back to the full combined stocks+ETFs universe.
if os.path.exists(os.path.join(_BASE_DIR, "us_universe_liquid.csv")):
    _CSV_CONFIGS = [("us_universe_liquid.csv", "US Liquid (Stocks+ETFs)", None)]
else:
    _CSV_CONFIGS = [("us_universe.csv", "US All (Stocks+ETFs)", None)]

# ── Known column name variants (handles whitespace, case, BOM differences) ───
_SYM_CANDIDATES = ["Symbol", "SYMBOL", "symbol", "Sym", "SYM",
                   "NSE Symbol", "NSESymbol"]
_SEC_CANDIDATES = ["Industry", "INDUSTRY", "industry", "Sector", "SECTOR",
                   "sector", "Ind", "IND", "IndustryName", "Industry Name"]
_SER_CANDIDATES = ["Series", "SERIES", "series"]


def _norm_cols(df):
    """Strip whitespace, BOM, and normalise column names in-place."""
    df.columns = [str(c).strip().lstrip("\ufeff").strip() for c in df.columns]
    return df


def _find_col(df_cols, candidates):
    """Return first matching column name (exact → case-insensitive)."""
    col_set   = set(df_cols)
    upper_map = {c.upper(): c for c in df_cols}
    for cand in candidates:
        if cand in col_set:
            return cand
        if cand.upper() in upper_map:
            return upper_map[cand.upper()]
    return None


SECTOR_STOCKS    = {}
SECTOR_MAP       = {}
UNIVERSE_SOURCES = []   # [(label, loaded_count, skipped_count, error_msg)]
_seen_symbols    = set()
UNIVERSE_ORDERED = []   # symbols in CSV (market-cap) order — for bounded scans


def _load_one_csv(filename, label, series_filter):
    """
    Load one NSE CSV. Returns (loaded, skipped, error_str).
    Handles: BOM, Windows line endings, mixed encoding, missing columns.
    """
    global SECTOR_STOCKS, SECTOR_MAP, _seen_symbols, UNIVERSE_ORDERED

    filepath = os.path.join(_BASE_DIR, filename)

    # ── 1. File existence check (absolute path) ───────────────────────────────
    if not os.path.exists(filepath):
        return 0, 0, f"File not found: {filepath}"

    # ── 2. Read with encoding fallback ────────────────────────────────────────
    df = None
    for enc in ("utf-8-sig", "utf-8", "latin-1", "cp1252"):
        try:
            df = pd.read_csv(filepath, encoding=enc,
                             on_bad_lines="skip", low_memory=False)
            _norm_cols(df)
            break
        except Exception:
            df = None

    if df is None or df.empty:
        return 0, 0, f"Could not parse {filename}"

    # ── 3. Find symbol column ─────────────────────────────────────────────────
    sym_col = _find_col(df.columns, _SYM_CANDIDATES)
    if sym_col is None:
        return 0, 0, (f"No symbol column in {filename}. "
                      f"Found: {list(df.columns[:5])}")

    # ── 4. Find sector column (optional) ─────────────────────────────────────
    sec_col = _find_col(df.columns, _SEC_CANDIDATES)   # None for EQUITY_L

    # ── 5. Apply series filter ────────────────────────────────────────────────
    if series_filter:
        ser_col = _find_col(df.columns, _SER_CANDIDATES)
        if ser_col:
            df = df[df[ser_col].astype(str).str.strip().str.upper()
                    .isin([s.upper() for s in series_filter])]

    # ── 6. Load rows ──────────────────────────────────────────────────────────
    loaded = skipped = 0
    for _, row in df.iterrows():
        sym = str(row[sym_col]).strip().upper()
        if not sym or sym in ("NAN", "SYMBOL", "SYM", ""):
            continue
        if sym in _seen_symbols:
            skipped += 1
            continue

        # Sector: from column if available, else label
        if sec_col:
            sector = str(row[sec_col]).strip()
            if not sector or sector.upper() in ("NAN", "INDUSTRY", "SECTOR", ""):
                sector = "Others"
        else:
            sector = label          # e.g. "NSE All Listed"

        if sector not in SECTOR_STOCKS:
            SECTOR_STOCKS[sector] = []
        SECTOR_STOCKS[sector].append(sym)
        SECTOR_MAP[sym] = sector
        _seen_symbols.add(sym)
        UNIVERSE_ORDERED.append(sym)
        loaded += 1

    return loaded, skipped, None     # None = no error


def debug_universe_load():
    """
    Returns a human-readable string showing what was loaded from each CSV,
    what files were scanned, and any errors. Use in app.py for diagnostics.
    """
    lines = [f"🔍 Universe load report — base dir: {_BASE_DIR}"]
    for filename, label, _ in _CSV_CONFIGS:
        fp = os.path.join(_BASE_DIR, filename)
        exists  = os.path.exists(fp)
        sz      = f"{os.path.getsize(fp):,} bytes" if exists else "—"
        status  = "✅ found" if exists else "❌ not found"
        lines.append(f"  {status}  {filename:40s}  {sz}")
    lines.append(f"\nLoaded sources:")
    for lbl, n, sk, err in UNIVERSE_SOURCES:
        if err:
            lines.append(f"  ❌ {lbl}: {err}")
        else:
            lines.append(f"  ✅ {lbl}: {n:,} new symbols, {sk:,} duplicates skipped")
    lines.append(f"\nTotal universe: {UNIVERSE_TOTAL:,} symbols across "
                 f"{len(SECTOR_STOCKS):,} sectors")
    return "\n".join(lines)


# ── Run loader at import time ─────────────────────────────────────────────────
_any_loaded = False
for _cfg in _CSV_CONFIGS:
    _fn, _lbl, _sf = _cfg
    _n, _sk, _err = _load_one_csv(_fn, _lbl, _sf)
    UNIVERSE_SOURCES.append((_lbl, _n, _sk, _err))
    if _n > 0:
        _any_loaded = True

if not _any_loaded:
    # Hardcoded fallback so app boots without any CSV
    SECTOR_STOCKS = {
        "Technology":             ["AAPL","MSFT","NVDA","GOOGL","META"],
        "Finance":                ["JPM","BAC","WFC","GS","V"],
        "Health Care":            ["UNH","JNJ","LLY","ABBV","MRK"],
        "Consumer Discretionary": ["AMZN","TSLA","HD","MCD","NKE"],
        "Energy":                 ["XOM","CVX","COP","SLB","EOG"],
        "Industrials":            ["CAT","BA","GE","HON","UPS"],
        "Broad Market ETF":       ["SPY","QQQ","IWM","DIA","VTI"],
    }
    for _sec, _stks in SECTOR_STOCKS.items():
        for _s in _stks:
            SECTOR_MAP[_s] = _sec
            UNIVERSE_ORDERED.append(_s)
    UNIVERSE_SOURCES = [("Fallback (no CSV found)", len(SECTOR_MAP), 0, None)]

# Total exposed for display
UNIVERSE_TOTAL = sum(len(v) for v in SECTOR_STOCKS.values())


# ETF symbol set — robust ETF detection regardless of how sectors are labelled
# (sector ETFs like XLK/SMH/GDX carry plain sector names, not "...ETF").
ETF_SYMBOLS = set()
try:
    _etf_fp = os.path.join(_BASE_DIR, "us_etfs.csv")
    if os.path.exists(_etf_fp):
        _edf = pd.read_csv(_etf_fp)
        _ecol = _find_col(_edf.columns, _SYM_CANDIDATES)
        if _ecol:
            ETF_SYMBOLS = set(_edf[_ecol].astype(str).str.strip().str.upper())
except Exception:
    ETF_SYMBOLS = set()


def _is_etf(sym):
    return sym in ETF_SYMBOLS or SECTOR_MAP.get(sym, "").endswith("ETF")


def get_scan_symbols(limit=None):
    """Bounded, most-liquid-first symbol list for the live scanners.

    The universe CSV is market-cap ordered, so the first N stocks are the most
    liquid. ALL ETFs are always included (there are few and they're liquid).
    This keeps each live scan small enough to avoid Yahoo rate-limiting on
    Streamlit Cloud — the root cause of empty scanner results.
    """
    lim = MAX_SCAN_SYMBOLS if limit is None else limit
    etfs, stocks = [], []
    for sym in UNIVERSE_ORDERED:
        (etfs if _is_etf(sym) else stocks).append(sym)
    if lim and lim > 0:
        stocks = stocks[:max(0, lim - len(etfs))]
    return etfs + stocks


def get_sector(symbol: str) -> str:
    clean_symbol = symbol.upper().replace(".NS", "").replace(".BO", "")
    return SECTOR_MAP.get(clean_symbol, "Others")


SECTOR_INDICES = {
    "Technology": "XLK", "Finance": "XLF", "Health Care": "XLV",
    "Energy": "XLE", "Industrials": "XLI",
    "Consumer Discretionary": "XLY", "Consumer Staples": "XLP",
    "Utilities": "XLU", "Basic Materials": "XLB",
    "Real Estate": "XLRE", "Telecommunications": "XLC",
}

TRACKED_INDICES = {
    "S&P 500": "^GSPC", "Nasdaq": "^IXIC",
    "Dow": "^DJI", "Russell 2000": "^RUT",
    "Nasdaq 100": "^NDX", "VIX": "^VIX"
}

# Fallback symbols for indices that Yahoo sometimes deprecates. If the primary
# symbol returns nothing, the resilient fetcher retries with these.
_INDEX_FALLBACKS = {
    "^GSPC": ["^GSPC", "SPY"],
    "^IXIC": ["^IXIC", "QQQ"],
    "^DJI":  ["^DJI", "DIA"],
    "^RUT":  ["^RUT", "IWM"],
    "^NDX":  ["^NDX", "QQQ"],
    "^VIX":  ["^VIX"],
}

# ─── Data Fetcher ──────────────────────────────────────────────────────────────
def _fetch_history(ticker, period="1y", interval="1d"):
    """Fetch OHLCV history for one ticker. Tries Ticker.history() first, then
    falls back to yf.download() which uses a different endpoint and is often
    more reliable on cloud hosts. Retries once on transient failure."""
    for _attempt in range(2):
        # Method 1: Ticker.history()  — auto_adjust=False gives ACTUAL prices
        # (matching what a trader sees on their chart), not div/split-adjusted.
        try:
            t = yf.Ticker(ticker)
            df = t.history(period=period, interval=interval, auto_adjust=False)
            if df is not None and not df.empty and "Close" in df.columns:
                return _normalize_ohlcv(df)
        except Exception:
            pass
        # Method 2: yf.download() fallback (different endpoint)
        try:
            df = yf.download(ticker, period=period, interval=interval,
                             auto_adjust=False, progress=False, threads=False)
            if df is not None and not df.empty:
                # download() may return multi-index columns for single ticker
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                if "Close" in df.columns:
                    return _normalize_ohlcv(df)
        except Exception:
            pass
        time.sleep(0.3)
    return None


def _normalize_ohlcv(df):
    """Standardise an OHLCV frame: guaranteed Open/High/Low/Close/Volume cols."""
    try:
        result = pd.DataFrame()
        result["Open"]   = df["Open"]   if "Open"   in df.columns else df["Close"]
        result["Close"]  = df["Close"]
        result["High"]   = df["High"]   if "High"   in df.columns else df["Close"]
        result["Low"]    = df["Low"]    if "Low"    in df.columns else df["Close"]
        result["Volume"] = df["Volume"] if "Volume" in df.columns else 0
        result = result.dropna(subset=["Close"]).ffill().bfill()
        return result if not result.empty else None
    except Exception:
        return None


def sanitize_ticker(sym):
    """Strips existing extensions to prevent SNOWMAN.NS.NS"""
    clean = str(sym).upper().strip()
    for suffix in [".NS", ".BO", ".NSE", ".BSE"]:
        if clean.endswith(suffix):
            clean = clean[:-len(suffix)]
    return clean


def _bulk_fetch_history(symbols, period="1y"):
    results = {}
    with ThreadPoolExecutor(max_workers=5) as executor:
        def fetch_single(sym):
            if sym.startswith("^"):
                return sym, _fetch_history(sym, period)
            df = _fetch_history(_yahoo(sym), period)
            return sym, df

        future_to_sym = {executor.submit(fetch_single, sym): sym for sym in symbols}
        for future in as_completed(future_to_sym):
            sym, df = future.result()
            if df is not None:
                results[sym] = df
    return results


# ─── Indicator Cache ──────────────────────────────────────────────────────────
_IND_CACHE = {}
_IND_CACHE_TS = {}
_CACHE_TTL = 900

# ==============================================================================
# FIX 1+4: Wilder's RSI and ATR with adjust=False and edge-case handling
# ==============================================================================
def compute_rsi_wilder(series, period=14):
    """Wilder's RSI. adjust=False matches TradingView/Zerodha exactly.
    Explicit 100/0 on all-gain/all-loss streaks instead of silent NaN."""
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rsi = pd.Series(np.nan, index=series.index, dtype=float)
    m100 = (avg_loss == 0) & (avg_gain > 0)          # pure uptrend → RSI 100
    m0   = (avg_gain == 0) & (avg_loss > 0)          # pure downtrend → RSI 0
    mn   = (avg_gain > 0) & (avg_loss > 0)
    rsi[m100] = 100.0
    rsi[m0]   = 0.0
    rs = avg_gain[mn] / avg_loss[mn]
    rsi[mn] = 100 - (100 / (1 + rs))
    return rsi


def compute_rsi(series, period=14):
    rsi = compute_rsi_wilder(series, period)
    val = rsi.iloc[-1]
    return round(float(val), 1) if not pd.isna(val) else None


def compute_atr_wilder(high, low, close, period=14):
    """True ATR with Wilder's EWM smoothing — matches Zerodha/TradingView.
    Returns (true_range_series, atr_series)."""
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    return tr, atr


# ─── Divergence Detection ─────────────────────────────────────────────────────
def detect_rsi_divergence(close, rsi_series, window=40):
    if len(close) < window or len(rsi_series) < window:
        return {"bullish_div": False, "bearish_div": False}
    c = close.iloc[-window:].values
    r = rsi_series.iloc[-window:].values
    troughs, _ = find_peaks(-c, distance=5)
    peaks, _   = find_peaks(c, distance=5)
    bullish = bearish = False
    if len(troughs) >= 2:
        if c[troughs[-1]] < c[troughs[-2]] and r[troughs[-1]] > r[troughs[-2]]:
            bullish = True
    if len(peaks) >= 2:
        if c[peaks[-1]] > c[peaks[-2]] and r[peaks[-1]] < r[peaks[-2]]:
            bearish = True
    return {"bullish_div": bullish, "bearish_div": bearish}


def detect_macd_divergence(close, macd_line, window=40):
    if len(close) < window or len(macd_line) < window:
        return {"bullish_div": False, "bearish_div": False}
    c = close.iloc[-window:].values
    m = macd_line.iloc[-window:].values
    troughs, _ = find_peaks(-c, distance=5)
    peaks, _   = find_peaks(c, distance=5)
    bullish = bearish = False
    if len(troughs) >= 2:
        if c[troughs[-1]] < c[troughs[-2]] and m[troughs[-1]] > m[troughs[-2]]:
            bullish = True
    if len(peaks) >= 2:
        if c[peaks[-1]] > c[peaks[-2]] and m[peaks[-1]] < m[peaks[-2]]:
            bearish = True
    return {"bullish_div": bullish, "bearish_div": bearish}


# ─── Chart Pattern Detection (unchanged from v11 — already 7+/10) ─────────────
def detect_price_patterns(high, low, close, vol, vol_avg):
    patterns = []
    if len(close) < 30:
        return patterns

    cmp    = float(close.iloc[-1])
    c_vals = close.values

    troughs, _ = find_peaks(-c_vals, distance=8, prominence=c_vals.std() * 0.3)
    peaks,   _ = find_peaks( c_vals, distance=8, prominence=c_vals.std() * 0.3)

    if len(close) >= 20:
        recent_h = high.iloc[-20:-1].max()
        recent_l = low.iloc[-20:-1].min()
        rng_pct  = (recent_h - recent_l) / recent_l
        if rng_pct < 0.10:
            if cmp > recent_h and float(vol.iloc[-1]) > vol_avg * 2.5:
                patterns.append("🚀 Vol Breakout")

    if len(close) >= 30:
        pole   = close.iloc[-30:-10]
        flag   = close.iloc[-10:-1]
        p_gain = (pole.max() - pole.min()) / (pole.min() + 1e-8)
        f_drop = (flag.max() - flag.min()) / (flag.max() + 1e-8)
        if p_gain > 0.08 and f_drop < 0.06 and flag.iloc[-1] < pole.max():
            if cmp > flag.max() and float(vol.iloc[-1]) > vol_avg * 2.0:
                patterns.append("🚩 Bull Flag Breakout")

    if len(troughs) >= 2:
        t1, t2 = troughs[-2], troughs[-1]
        p1, p2 = c_vals[t1], c_vals[t2]
        depth_ok    = abs(p1 - p2) / (p1 + 1e-8) < 0.08
        price_ok    = p2 * 1.00 < cmp < p2 * 1.12
        vol_confirm = float(vol.iloc[-1]) > vol_avg * 1.2
        if depth_ok and price_ok and vol_confirm:
            patterns.append("📉 Double Bottom")

    if len(peaks) >= 2:
        p1_idx, p2_idx = peaks[-2], peaks[-1]
        v1, v2 = c_vals[p1_idx], c_vals[p2_idx]
        if abs(v1 - v2) / (v1 + 1e-8) < 0.08 and v2 * 0.88 < cmp < v2 * 0.99:
            patterns.append("📈 Double Top")

    if len(peaks) >= 3 and len(troughs) >= 2:
        p1, p2, p3 = c_vals[peaks[-3]], c_vals[peaks[-2]], c_vals[peaks[-1]]
        head_valid = p2 > p1 and p2 > p3 and abs(p1 - p3) / (p1 + 1e-8) < 0.06
        if head_valid:
            neckline = (c_vals[troughs[-2]] + c_vals[troughs[-1]]) / 2
            if cmp < neckline * 0.99 and float(vol.iloc[-1]) > vol_avg * 1.3:
                patterns.append("🏔️ Head & Shoulders (Top)")

    if len(troughs) >= 3 and len(peaks) >= 2:
        t1, t2, t3 = c_vals[troughs[-3]], c_vals[troughs[-2]], c_vals[troughs[-1]]
        head_valid = t2 < t1 and t2 < t3 and abs(t1 - t3) / (t1 + 1e-8) < 0.06
        if head_valid:
            neckline = (c_vals[peaks[-2]] + c_vals[peaks[-1]]) / 2
            if cmp > neckline * 1.01 and float(vol.iloc[-1]) > vol_avg * 1.3:
                patterns.append("🛤️ Inverse H&S (Bottom)")

    if len(close) >= 60:
        cup_window = close.iloc[-60:-10]
        handle     = close.iloc[-10:]
        cup_left   = float(cup_window.iloc[0])
        cup_right  = float(cup_window.iloc[-1])
        cup_base   = float(cup_window.min())
        cup_depth  = (cup_left - cup_base) / (cup_left + 1e-8)
        rim_match  = abs(cup_left - cup_right) / (cup_left + 1e-8)
        handle_ret = (float(handle.max()) - float(handle.min())) / (float(handle.max()) + 1e-8)
        breakout   = cmp > float(handle.max()) * 0.995
        if (0.10 < cup_depth < 0.40 and rim_match < 0.06 and
                handle_ret < 0.08 and breakout and float(vol.iloc[-1]) > vol_avg * 1.5):
            patterns.append("☕ Cup & Handle Breakout")

    return patterns


# ─── Candlestick Detection (unchanged from v11 — already 8/10) ────────────────
def detect_candlesticks(open_p, high, low, close):
    candles = []
    if len(close) < 5:
        return candles

    def _candle(i):
        o, h, l, c = float(open_p.iloc[i]), float(high.iloc[i]), float(low.iloc[i]), float(close.iloc[i])
        body = abs(c - o); rng = h - l
        if rng < 1e-8: return None
        upper_wick = h - max(o, c); lower_wick = min(o, c) - l
        bullish = c > o
        return dict(o=o, h=h, l=l, c=c, body=body, rng=rng,
                    upper_wick=upper_wick, lower_wick=lower_wick, bullish=bullish)

    c0 = _candle(-1); c1 = _candle(-2)
    c2 = _candle(-3) if len(close) >= 3 else None
    if not c0 or not c1:
        return candles

    if (c0["lower_wick"] >= c0["rng"] * 0.55 and c0["upper_wick"] <= c0["rng"] * 0.15 and c0["body"] >= c0["rng"] * 0.05):
        candles.append("🔨 Bullish Hammer")
    if (c0["upper_wick"] >= c0["rng"] * 0.55 and c0["lower_wick"] <= c0["rng"] * 0.15 and c0["body"] >= c0["rng"] * 0.05 and not c0["bullish"]):
        candles.append("💫 Shooting Star")
    if c0["body"] <= c0["rng"] * 0.07:
        candles.append("〰️ Doji (Indecision)")
    if (not c1["bullish"] and c0["bullish"] and c0["o"] <= c1["c"] and c0["c"] >= c1["o"] and c0["body"] > c1["body"] * 1.0):
        candles.append("🟩 Bullish Engulfing")
    if (c1["bullish"] and not c0["bullish"] and c0["o"] >= c1["c"] and c0["c"] <= c1["o"] and c0["body"] > c1["body"] * 1.0):
        candles.append("🟥 Bearish Engulfing")
    if (not c1["bullish"] and c0["bullish"] and c0["o"] > c1["c"] and c0["c"] < c1["o"] and c0["body"] < c1["body"] * 0.5):
        candles.append("🟢 Bullish Harami")
    if (not c1["bullish"] and c0["bullish"] and c0["o"] < c1["l"] and c0["c"] > (c1["o"] + c1["c"]) / 2 and c0["c"] < c1["o"]):
        candles.append("🔆 Piercing Line")

    if c2:
        if (not c2["bullish"] and c2["body"] >= c2["rng"] * 0.5 and c1["body"] <= c1["rng"] * 0.3 and
                c0["bullish"] and c0["body"] >= c0["rng"] * 0.5 and c0["c"] > (c2["o"] + c2["c"]) / 2):
            candles.append("🌅 Morning Star")
        if (c2["bullish"] and c2["body"] >= c2["rng"] * 0.5 and c1["body"] <= c1["rng"] * 0.3 and
                not c0["bullish"] and c0["body"] >= c0["rng"] * 0.5 and c0["c"] < (c2["o"] + c2["c"]) / 2):
            candles.append("🌆 Evening Star")
        if (c2["bullish"] and c1["bullish"] and c0["bullish"] and
                c1["o"] > c2["o"] and c0["o"] > c1["o"] and c1["c"] > c2["c"] and c0["c"] > c1["c"] and
                c0["body"] >= c0["rng"] * 0.5 and c1["body"] >= c1["rng"] * 0.5):
            candles.append("🪖 Three White Soldiers")
        if (not c2["bullish"] and not c1["bullish"] and not c0["bullish"] and
                c1["o"] < c2["o"] and c0["o"] < c1["o"] and c1["c"] < c2["c"] and c0["c"] < c1["c"] and
                c0["body"] >= c0["rng"] * 0.5 and c1["body"] >= c1["rng"] * 0.5):
            candles.append("🦅 Three Black Crows")

    return candles


# ─── Market Regime Detection ──────────────────────────────────────────────────
_market_regime_cache = {"ts": 0, "data": None}


def _fetch_index_history(symbol, period="1y"):
    """Fetch a single index (^-prefixed) with maximum resilience.
    Indices on Yahoo are flaky — try the primary symbol then any known
    fallbacks, each with Ticker.history then yf.download. Returns a
    normalised OHLC frame or None."""
    candidates = _INDEX_FALLBACKS.get(symbol, [symbol])
    if symbol not in candidates:
        candidates = [symbol] + candidates

    for sym in candidates:
        for _attempt in range(2):
            # Method 1: Ticker.history
            try:
                t = yf.Ticker(sym)
                df = t.history(period=period, interval="1d", auto_adjust=False)
                if df is not None and not df.empty and "Close" in df.columns:
                    out = _normalize_ohlcv(df)
                    if out is not None and len(out) >= 2:
                        return out
            except Exception:
                pass
            # Method 2: yf.download (different endpoint)
            try:
                df = yf.download(sym, period=period, interval="1d",
                                 auto_adjust=False, progress=False, threads=False)
                if df is not None and not df.empty:
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    if "Close" in df.columns:
                        out = _normalize_ohlcv(df)
                        if out is not None and len(out) >= 2:
                            return out
            except Exception:
                pass
            time.sleep(0.3)
    return None


# ==============================================================================
# RELATIVE STRENGTH (RS) vs NIFTY  —  IBD / Minervini style leadership ranking
# ==============================================================================
# RS answers: "Is this stock a leader or a laggard versus the broad market?"
# Two layers:
#   1. RS Ratio  — a weighted blend of the stock's return vs Nifty's return over
#                  multiple lookbacks (recent weighted heaviest). >1.0 = leading.
#   2. RS Rating — that ratio converted to a 1-99 PERCENTILE across the scanned
#                  universe (IBD-style: 80+ = strong leader, <30 = laggard).
#
# A high-RS stock in a VCP base near a pivot is the classic Minervini long.
# ==============================================================================

_NIFTY_BENCH_CACHE = {"ts": 0, "data": None}


def _get_nifty_benchmark():
    """Return Nifty's trailing returns over standard lookbacks (cached 15 min).
    Returns dict {'21': r, '63': r, '126': r, '252': r} of % returns, or None."""
    now = time.time()
    if _NIFTY_BENCH_CACHE["data"] and (now - _NIFTY_BENCH_CACHE["ts"]) < _CACHE_TTL:
        return _NIFTY_BENCH_CACHE["data"]
    df = _fetch_index_history(BENCH_SYM, period="1y")
    if df is None or df.empty or len(df) < 30:
        return None
    closes = df["Close"].dropna().values.astype(float)
    bench = {}
    for lb in (21, 63, 126, 252):
        if len(closes) > lb:
            past = closes[-lb - 1]
            bench[str(lb)] = (closes[-1] / past - 1) * 100 if past > 0 else 0.0
        else:
            # Not enough history for this window — use the longest available
            past = closes[0]
            bench[str(lb)] = (closes[-1] / past - 1) * 100 if past > 0 else 0.0
    _NIFTY_BENCH_CACHE["data"] = bench
    _NIFTY_BENCH_CACHE["ts"] = now
    return bench


def compute_relative_strength(close, bench=None):
    """Compute a stock's RS ratio versus Nifty.

    Uses IBD-style weighting: the most recent quarter counts double.
    RS ratio > 1.0 means the stock is OUTPERFORMING Nifty; < 1.0 underperforming.

    Returns dict: {rs_ratio, rs_line, outperforming, periods:{...}} or None.
    """
    if bench is None:
        bench = _get_nifty_benchmark()
    if bench is None:
        return None
    if close is None or len(close) < 30:
        return None

    c = close.values.astype(float) if hasattr(close, "values") else np.asarray(close, float)
    c = c[~np.isnan(c)]
    if len(c) < 30:
        return None

    # Stock returns over the same lookbacks
    periods = {}
    weights = {"21": 0.4, "63": 0.2, "126": 0.2, "252": 0.2}   # recent weighted 2x
    rs_components = []
    total_w = 0.0
    for lb_str, w in weights.items():
        lb = int(lb_str)
        if len(c) > lb:
            past = c[-lb - 1]
        else:
            past = c[0]
        stock_ret = (c[-1] / past - 1) * 100 if past > 0 else 0.0
        nifty_ret = bench.get(lb_str, 0.0)
        periods[lb_str] = {"stock": round(stock_ret, 1), "nifty": round(nifty_ret, 1)}
        # RS component = (1+stock%) / (1+nifty%) — ratio of growth factors
        sf = 1 + stock_ret / 100.0
        nf = 1 + nifty_ret / 100.0
        if nf > 0:
            rs_components.append((sf / nf) * w)
            total_w += w

    if total_w == 0:
        return None
    rs_ratio = sum(rs_components) / total_w

    return {
        "rs_ratio": round(rs_ratio, 3),
        "outperforming": rs_ratio > 1.0,
        "periods": periods,
    }


def _rs_ratio_to_rating(ratio, all_ratios):
    """Convert an RS ratio to a 1-99 percentile rating within the universe."""
    if not all_ratios:
        return None
    below = sum(1 for r in all_ratios if r < ratio)
    pct = below / len(all_ratios) * 100
    return max(1, min(99, int(round(pct))))


def get_market_regime():
    now = time.time()
    if _market_regime_cache["data"] and (now - _market_regime_cache["ts"]) < _CACHE_TTL:
        return _market_regime_cache["data"]

    indices_data = {}
    bulk_data = {}

    # Fetch each index individually with the resilient fetcher. Doing them one
    # by one (not bulk) means one failing index never wipes out the rest.
    for name, symbol in TRACKED_INDICES.items():
        df = _fetch_index_history(symbol, period="1y")
        if df is not None and len(df) >= 2:
            bulk_data[symbol] = df
            current = float(df["Close"].iloc[-1])
            prev    = float(df["Close"].iloc[-2])
            chg     = round((current / prev - 1) * 100, 2)
            indices_data[name] = {"price": round(current, 2), "chg_pct": chg}
        elif df is not None and len(df) == 1:
            bulk_data[symbol] = df
            current = float(df["Close"].iloc[-1])
            indices_data[name] = {"price": round(current, 2), "chg_pct": 0.0}

    nifty = indices_data.get(BENCH_NAME, {})
    nifty_close = nifty.get("price")
    regime, trend, nifty_rsi = "Unknown", "Sideways", None
    support, resistance, conf = None, None, 50

    if nifty_close and BENCH_SYM in bulk_data:
        df = bulk_data[BENCH_SYM]
        if len(df) >= 50:
            close  = df["Close"]
            ema20  = float(close.ewm(span=20,  adjust=False).mean().iloc[-1])
            ema50  = float(close.ewm(span=50,  adjust=False).mean().iloc[-1])
            ema200 = float(close.ewm(span=200, adjust=False).mean().iloc[-1]) if len(close) >= 200 else None
            nifty_rsi  = compute_rsi(close)
            support    = float(df["Low"].rolling(20).min().iloc[-1])
            resistance = float(df["High"].rolling(20).max().iloc[-1])

            if ema200 and nifty_close > ema200:
                if nifty_close > ema20 > ema50: regime, trend = "Strong Bull", "Uptrend"
                elif nifty_close > ema20: regime, trend = "Bull", "Uptrend"
                else: regime, trend = "Bull Pullback", "Pullback"
            elif ema200 and nifty_close < ema200:
                if nifty_close < ema20 < ema50: regime, trend = "Strong Bear", "Downtrend"
                elif nifty_close < ema20: regime, trend = "Bear", "Downtrend"
                else: regime, trend = "Bear Rally", "Relief Rally"
            else:
                regime, trend = "Neutral", "Sideways"

            if regime in ("Strong Bull", "Strong Bear"): conf = 85
            elif regime in ("Bull", "Bear"): conf = 70
            elif regime in ("Bull Pullback", "Bear Rally"): conf = 55

    if nifty_rsi:
        if nifty_rsi > 70: conf = min(95, conf + 15)
        elif nifty_rsi < 40: conf = max(20, conf - 20)

    risk = "Neutral"
    if nifty_rsi:
        if nifty_rsi > 70: risk = "High Momentum (Power Zone)"
        elif nifty_rsi > 60: risk = "Building Momentum"
        elif nifty_rsi < 40: risk = "High Risk (Downtrend/Bleeding)"

    result = {
        "regime": regime, "trend": trend, "nifty_close": nifty_close,
        "nifty_rsi": nifty_rsi, "risk_level": risk, "indices": indices_data,
        "support": support, "resistance": resistance, "confidence": conf,
        "indices_ok": len(indices_data) > 0,
    }
    # Only cache a GOOD result (with at least some index data). If everything
    # failed (transient Yahoo rate-limit), don't poison the 10-min cache with
    # an empty result — let the next call retry instead.
    if indices_data:
        _market_regime_cache["data"] = result
        _market_regime_cache["ts"] = now
    return result


# ==============================================================================
# TECHNICAL INDICATORS — v12 with all fixes
# ==============================================================================
def _compute_indicators_raw(symbol, period="1y", prefetched_df=None):
    df = prefetched_df
    if df is None:
        df = _fetch_history(_yahoo(symbol), period=period, interval="1d")

    # Minimum bars: need ~20 for the rolling-20 indicators (BB, S/R, vol avg).
    # Newly listed stocks with 20-49 bars get computed but flagged as limited.
    # Below 20 bars there isn't enough to compute anything reliable.
    if df is None or len(df) < 20:
        return None
    _limited_history = len(df) < 50   # newly listed → some long-period signals N/A

    open_p = df["Open"]; high = df["High"]; low = df["Low"]
    close = df["Close"]; vol = df["Volume"]
    cmp = float(close.iloc[-1])
    if pd.isna(cmp) or cmp <= 0:
        return None

    # ── FIX 1: RSI with adjust=False + edge case ──────────────────────────────
    rsi_series = compute_rsi_wilder(close, 14)
    rsi = round(float(rsi_series.iloc[-1]), 1) if not pd.isna(rsi_series.iloc[-1]) else None

    # ── FIX 7: EMAs with adjust=False + slope detection + EMA200 restored ────
    ema9_s   = close.ewm(span=9,   adjust=False).mean()
    ema21_s  = close.ewm(span=21,  adjust=False).mean()
    ema50_s  = close.ewm(span=50,  adjust=False).mean()
    ema200_s = close.ewm(span=200, adjust=False).mean() if len(close) >= 100 else None

    ema9   = float(ema9_s.iloc[-1])
    ema21  = float(ema21_s.iloc[-1])
    ema50  = float(ema50_s.iloc[-1])
    ema200 = float(ema200_s.iloc[-1]) if ema200_s is not None else None

    # Slope over last 3 bars — momentum direction of the EMA itself
    ema9_slope  = float(ema9_s.iloc[-1] - ema9_s.iloc[-4])  if len(ema9_s)  >= 4 else 0.0
    ema21_slope = float(ema21_s.iloc[-1] - ema21_s.iloc[-4]) if len(ema21_s) >= 4 else 0.0
    ema_rising      = ema9_slope > 0 and ema21_slope > 0
    ema_flattening  = abs(ema9_slope) < (cmp * 0.001)   # <0.1% of price over 3 bars

    # ── FIX 2: MACD — adjust=False, single-pass crossover, histogram ─────────
    macd_fast   = close.ewm(span=12, adjust=False).mean()
    macd_slow   = close.ewm(span=26, adjust=False).mean()
    macd_line   = macd_fast - macd_slow
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    histogram   = macd_line - signal_line

    macd_bullish = macd_bearish = False
    if len(macd_line) >= 3:
        prev_diff = float(macd_line.iloc[-2]) - float(signal_line.iloc[-2])
        curr_diff = float(macd_line.iloc[-1]) - float(signal_line.iloc[-1])
        if prev_diff <= 0 and curr_diff > 0:
            macd_bullish = True            # fresh bullish cross — mutually exclusive
        elif prev_diff >= 0 and curr_diff < 0:
            macd_bearish = True            # fresh bearish cross

    hist_val   = round(float(histogram.iloc[-1]), 4)
    hist_slope = float(histogram.iloc[-1] - histogram.iloc[-2]) if len(histogram) >= 2 else 0.0
    macd_hist_expanding   = hist_val > 0 and hist_slope > 0   # bull momentum building
    macd_hist_contracting = hist_val > 0 and hist_slope < 0   # bull momentum fading

    rsi_div  = detect_rsi_divergence(close, rsi_series)
    macd_div = detect_macd_divergence(close, macd_line)

    # ── FIX 3: Bollinger — clamped bb_pos + bandwidth + squeeze ──────────────
    bb_sma = float(close.rolling(20).mean().iloc[-1])
    bb_std = float(close.rolling(20).std().iloc[-1])
    if pd.isna(bb_sma): bb_sma = float(close.mean())
    if pd.isna(bb_std): bb_std = float(close.std()) if len(close) > 1 else 0.0
    bb_upper, bb_lower = bb_sma + 2 * bb_std, bb_sma - 2 * bb_std
    bb_range = bb_upper - bb_lower
    bb_pos = round(max(0.0, min(1.0, (cmp - bb_lower) / bb_range)), 2) if bb_range > 0 else 0.5
    # Breakout flags — replace info lost by clamping
    bb_breakout_up   = cmp > bb_upper
    bb_breakout_down = cmp < bb_lower

    bb_bandwidth = round(bb_range / bb_sma, 4) if bb_sma > 0 else None
    bb_squeeze   = False
    if bb_bandwidth is not None and len(close) >= 40:
        bw_series  = (close.rolling(20).std() * 4) / close.rolling(20).mean()
        bw_avg     = float(bw_series.rolling(20).mean().iloc[-1])
        if not pd.isna(bw_avg) and bw_avg > 0:
            bb_squeeze = bb_bandwidth < bw_avg * 0.75   # 25% below recent avg

    # ── FIX 4: Wilder ATR ──────────────────────────────────────────────────────
    tr, atr_series = compute_atr_wilder(high, low, close, 14)
    atr = float(atr_series.iloc[-1]) if not pd.isna(atr_series.iloc[-1]) \
        else float(high.iloc[-1] - low.iloc[-1])

    # ── Volume & Liquidity (FIX 10: soft gate, not silent None) ──────────────
    vol_avg   = float(vol.rolling(20).mean().iloc[-1]) if len(vol) >= 20 else float(vol.mean())
    vol_ratio = float(vol.iloc[-1]) / vol_avg if vol_avg > 0 else 1.0
    avg_turnover = vol_avg * cmp
    liquidity_ok = avg_turnover >= 10_000_000   # $10M daily turnover

    # ── 20-day S/R ─────────────────────────────────────────────────────────────
    _sr_window = min(20, len(close))
    support    = float(low.rolling(_sr_window).min().iloc[-1])
    resistance = float(high.rolling(_sr_window).max().iloc[-1])
    if pd.isna(support):    support = float(low.min())
    if pd.isna(resistance): resistance = float(high.max())
    high52, low52 = float(high.max()), float(low.min())

    # ── Trend with slope guard ────────────────────────────────────────────────
    trend = "Sideways"
    if cmp > ema9 > ema21:
        trend = "Strong Uptrend"
        if ema_flattening:
            trend = "Uptrend (Momentum Fading)"
    elif cmp > ema21:
        trend = "Uptrend" if ema_rising else "Recovery"
    elif cmp < ema9 < ema21:
        trend = "Strong Downtrend"
    elif cmp < ema21:
        trend = "Downtrend"

    # ── FIX 8: Fibonacci — swing-peak based via scipy ─────────────────────────
    c_vals = close.values
    swing_peaks,   _ = find_peaks( c_vals, distance=8, prominence=c_vals.std() * 0.3)
    swing_troughs, _ = find_peaks(-c_vals, distance=8, prominence=c_vals.std() * 0.3)

    fib_h = float(c_vals[swing_peaks[-1]])   if len(swing_peaks)   >= 1 else float(high.tail(60).max())
    fib_l = float(c_vals[swing_troughs[-1]]) if len(swing_troughs) >= 1 else float(low.tail(60).min())
    if fib_h < fib_l:                        # latest trough is above latest peak → swap
        fib_h, fib_l = fib_l, fib_h
    if (fib_h - fib_l) / (fib_l + 1e-8) < 0.02:   # degenerate swing → widen to 60-bar
        fib_h, fib_l = float(high.tail(60).max()), float(low.tail(60).min())

    fib_d   = fib_h - fib_l
    fib_236 = round(fib_h - fib_d * 0.236, 2)
    fib_382 = round(fib_h - fib_d * 0.382, 2)
    fib_500 = round(fib_h - fib_d * 0.500, 2)
    fib_618 = round(fib_h - fib_d * 0.618, 2)

    chart_patterns = detect_price_patterns(high, low, close, vol, vol_avg)
    candlesticks   = detect_candlesticks(open_p, high, low, close)

    # ── FIX 5: Supertrend — numpy loop, Wilder ATR(10), multiplier 2.5 ───────
    supertrend, supertrend_bullish = None, None
    try:
        st_period = 10
        st_mult   = 2.5     # 2.5 for NSE 5-15 day swing; 3.0 = positional
        _, atr_st_series = compute_atr_wilder(high, low, close, st_period)
        atr_st = atr_st_series.values
        hl2    = ((high + low) / 2).values
        c_arr  = close.values
        n      = len(c_arr)

        st_arr  = np.full(n, np.nan)
        dir_arr = np.ones(n, dtype=int)

        for i in range(n):
            if i < st_period or np.isnan(atr_st[i]):
                st_arr[i]  = c_arr[i]
                dir_arr[i] = 1
                continue
            upper = hl2[i] + st_mult * atr_st[i]
            lower = hl2[i] - st_mult * atr_st[i]
            if dir_arr[i - 1] == 1:                     # was bullish
                if c_arr[i] < st_arr[i - 1]:            # crossed below → flip
                    dir_arr[i] = -1
                    st_arr[i]  = upper
                else:
                    dir_arr[i] = 1
                    st_arr[i]  = max(lower, st_arr[i - 1])   # trail up only
            else:                                       # was bearish
                if c_arr[i] > st_arr[i - 1]:            # crossed above → flip
                    dir_arr[i] = 1
                    st_arr[i]  = lower
                else:
                    dir_arr[i] = -1
                    st_arr[i]  = min(upper, st_arr[i - 1])   # trail down only

        supertrend         = round(float(st_arr[-1]), 2)
        supertrend_bullish = bool(dir_arr[-1] == 1)
    except Exception:
        pass

    # ── FIX 6: VWAP — 20-day rolling, anchored to typical price ──────────────
    vwap, price_vs_vwap = None, None
    try:
        typical = (high + low + close) / 3
        roll_n  = min(20, len(close))
        cum_tpv = (typical * vol).rolling(roll_n).sum()
        cum_v   = vol.rolling(roll_n).sum().replace(0, np.nan)
        vwap_s  = cum_tpv / cum_v
        v = float(vwap_s.iloc[-1])
        if not pd.isna(v) and v > 0:
            vwap = round(v, 2)
            price_vs_vwap = round((cmp - vwap) / vwap * 100, 2)
    except Exception:
        pass

    # ── Bull / Bear Trap detection (v4 — ATR-normalised, RSI-at-peak) ────────
    # Pull cached market regime so the regime-context factor activates live.
    try:
        _mkt_regime = get_market_regime()
    except Exception:
        _mkt_regime = None
    traps = detect_trap_signals(
        close, high, low, vol, vol_avg, rsi_series,
        supertrend_bullish, resistance, support,
        candlesticks, atr, high52, low52, window=15,
        market_regime=_mkt_regime
    )

    # ── Smart Money Concepts (FVG, OB, Liquidity, Premium/Discount, Displacement) ─
    try:
        smc = compute_smc(open_p, high, low, close, vol, atr)
    except Exception:
        smc = {}

    # ── VCP — Volatility Contraction Pattern (Minervini) ─────────────────────
    try:
        vcp = detect_vcp(close, high, low, vol, atr, lookback=80)
    except Exception:
        vcp = {"is_vcp": False, "vcp_ready": False, "contractions": [],
               "pivot": None, "quality": None, "detail": ""}

    # ── Relative Strength vs Nifty ───────────────────────────────────────────
    try:
        rs = compute_relative_strength(close)
    except Exception:
        rs = None

    return {
        "symbol": symbol, "cmp": round(cmp, 2), "rsi": rsi,
        "limited_history": _limited_history, "bars": len(df),

        # EMAs — v12 adds slope flags + ema200
        "ema9": round(ema9, 2), "ema21": round(ema21, 2), "ema50": round(ema50, 2),
        "ema200": round(ema200, 2) if ema200 else None,
        "ema_rising": ema_rising, "ema_flattening": ema_flattening,
        # Legacy aliases for app.py
        "ema20": round(ema9, 2), "ema50_alias": round(ema21, 2),

        # MACD — v12 adds histogram intelligence
        "macd_bullish": macd_bullish, "macd_bearish": macd_bearish,
        "macd_histogram": hist_val,
        "macd_hist_expanding": macd_hist_expanding,
        "macd_hist_contracting": macd_hist_contracting,

        "rsi_divergence": rsi_div, "macd_divergence": macd_div,

        # BB — v12 adds bandwidth/squeeze/breakout flags
        "bb_upper": round(bb_upper, 2), "bb_lower": round(bb_lower, 2),
        "bb_sma": round(bb_sma, 2), "bb_pos": bb_pos,
        "bb_bandwidth": bb_bandwidth, "bb_squeeze": bb_squeeze,
        "bb_breakout_up": bb_breakout_up, "bb_breakout_down": bb_breakout_down,

        "atr": round(atr, 2), "vol_ratio": round(vol_ratio, 2),
        "support": round(support, 2), "resistance": round(resistance, 2),
        "high52": round(high52, 2), "low52": round(low52, 2),
        "trend": trend,
        "fib_236": fib_236, "fib_382": fib_382, "fib_500": fib_500, "fib_618": fib_618,
        "supertrend": supertrend, "supertrend_bullish": supertrend_bullish,
        "vwap": vwap, "price_vs_vwap": price_vs_vwap,
        "patterns": chart_patterns, "candlesticks": candlesticks,
        "avg_turnover": avg_turnover, "atr_pct": (atr / cmp) if cmp > 0 else 0,
        "liquidity_ok": liquidity_ok,   # FIX 10: visible flag, not silent None
        # Trap signals
        "bull_trap": traps["bull_trap"],
        "bear_trap": traps["bear_trap"],
        "bull_trap_conf": traps["bull_trap_conf"],
        "bear_trap_conf": traps["bear_trap_conf"],
        "bull_trap_detail": traps["bull_trap_detail"],
        "bear_trap_detail": traps["bear_trap_detail"],
        # VCP — Volatility Contraction Pattern
        "vcp": vcp.get("is_vcp", False),
        "vcp_ready": vcp.get("vcp_ready", False),
        "vcp_quality": vcp.get("quality"),
        "vcp_pivot": vcp.get("pivot"),
        "vcp_pivot_dist": vcp.get("pivot_distance_pct"),
        "vcp_contractions": vcp.get("contractions", []),
        "vcp_detail": vcp.get("detail", ""),
        # Relative Strength vs Nifty
        "rs_ratio": rs.get("rs_ratio") if rs else None,
        "rs_outperforming": rs.get("outperforming") if rs else None,
        "rs_periods": rs.get("periods") if rs else None,
        # Smart Money Concepts — merged via **smc
        **smc,
    }


def compute_indicators(symbol, period="1y", prefetched_df=None):
    now = time.time()
    key = f"{symbol}_{period}"
    if key in _IND_CACHE and (now - _IND_CACHE_TS.get(key, 0)) < _CACHE_TTL:
        return _IND_CACHE[key]
    result = _compute_indicators_raw(symbol, period, prefetched_df)
    if result is not None:
        _IND_CACHE[key] = result
        _IND_CACHE_TS[key] = now
    return result


# ==============================================================================
# FIX 9: UNIFIED RISK ENGINE — single source of truth, now used EVERYWHERE
# ==============================================================================
def _calc_risk_params(cmp, atr, resistance, buy_at=None, pct=None,
                      supertrend_val=None, supertrend_bullish=None,
                      action="HOLD"):
    """
    HOLD/AVERAGE/WATCH (existing positions):
      Target = max(20d resistance, cmp + 2.5*ATR)
      SL     = cmp - 2.0*ATR, raised to supertrend if bullish & higher
    PICK (fresh entries — sector picks AND universe scanner):
      Target = max(cmp*1.10, cmp + 2.5*ATR)
      SL     = cmp - 1.25*ATR
    SELL:
      Target = cmp (exit now), SL = re-entry zone
    """
    if action == "PICK":
        stop_loss = round(cmp - 1.25 * atr, 2)
        target    = round(max(cmp * 1.10, cmp + 2.5 * atr), 2)
    elif action in ("HOLD", "AVERAGE", "WATCH"):
        stop_loss = round(cmp - 2.0 * atr, 2)
        if supertrend_bullish and supertrend_val and supertrend_val > stop_loss:
            stop_loss = round(supertrend_val, 2)
        target = round(max(resistance, cmp + 2.5 * atr), 2)
    else:   # SELL
        stop_loss = round(cmp - 2.0 * atr, 2)
        target    = round(cmp, 2)

    risk   = cmp - stop_loss
    reward = target - cmp
    rr = round(reward / risk, 2) if risk > 0.01 else None
    return target, stop_loss, rr


# ─── Expert Signal Engine ─────────────────────────────────────────────────────
def generate_signals(trades_df):
    signals = []
    open_trades = trades_df[trades_df["status"] == "Open"].copy()
    if open_trades.empty:
        return signals

    market = get_market_regime()
    is_bear = market["regime"] in ("Strong Bear", "Bear")

    unique_symbols = open_trades["stock"].unique().tolist()
    bulk_data = _bulk_fetch_history(unique_symbols, period="1y")

    for _, row in open_trades.iterrows():
        symbol, buy_at, qty, tid = row["stock"], row["buy_at"], row["quantity"], row["id"]

        df  = bulk_data.get(symbol)
        ind = compute_indicators(symbol, period="1y", prefetched_df=df)

        if ind is None:
            signals.append({
                "id": tid, "stock": symbol, "sector": get_sector(symbol),
                "action": "⚪ WATCH",
                "reason": "No usable data — either the NSE symbol is wrong, or it's "
                          "newly listed with under 20 trading days of history",
                "strength": 0, "cmp": None, "rsi": None, "pct_from_buy": None,
                "target": None, "stop_loss": None, "avg_price": None,
                "new_avg": None, "new_sl": None, "macd_signal": "—",
                "bb_position": "—", "trend": "—", "support": None,
                "resistance": None, "risk_reward": None, "buy_at": buy_at,
                "quantity": qty, "market_regime": market["regime"],
                "divergence": "—", "supertrend": "—", "vwap": None, "fib_levels": {},
            })
            continue

        cmp, rsi = ind["cmp"], ind["rsi"]
        ema9, ema21, ema50 = ind["ema9"], ind["ema21"], ind["ema50"]
        support, resistance = ind["support"], ind["resistance"]
        atr   = ind["atr"]
        trend = ind["trend"]
        macd_bull, macd_bear = ind["macd_bullish"], ind["macd_bearish"]
        hist_fading          = ind.get("macd_hist_contracting", False)
        bb_pos               = ind["bb_pos"]
        bb_breakout_up       = ind.get("bb_breakout_up", False)
        rsi_div, macd_div    = ind["rsi_divergence"], ind["macd_divergence"]
        st_bullish, st_val   = ind.get("supertrend_bullish"), ind.get("supertrend")
        vwap        = ind.get("vwap")
        pv_vwap     = ind.get("price_vs_vwap")
        patterns    = ind.get("patterns", [])
        candles     = ind.get("candlesticks", [])
        bb_squeeze  = ind.get("bb_squeeze", False)
        ema_fading  = ind.get("ema_flattening", False)
        bull_trap   = ind.get("bull_trap", False)
        bear_trap   = ind.get("bear_trap", False)
        bt_conf     = ind.get("bull_trap_conf", 0)
        bt_detail   = ind.get("bull_trap_detail", "")
        brt_conf    = ind.get("bear_trap_conf", 0)
        brt_detail  = ind.get("bear_trap_detail", "")

        pct      = round((cmp - buy_at) / buy_at * 100, 2)
        near52h  = cmp >= ind["high52"] * 0.97
        nifty_chg = market.get("indices", {}).get("Nifty 50", {}).get("chg_pct", 0)
        stock_rs_strong = pct > nifty_chg

        # ── SELL Triggers ─────────────────────────────────────────────────────
        sell = []
        if rsi and rsi >= 75: sell.append(f"RSI overbought ({rsi})")
        if rsi and rsi >= 70 and near52h: sell.append("Near 52w high")
        atr_trail = round(cmp - 2.0 * atr, 2)
        effective_trail = round(max(atr_trail, st_val), 2) if (st_bullish and st_val) else atr_trail
        if cmp < effective_trail: sell.append(f"Below Trail Stop (${effective_trail})")
        elif not st_bullish: sell.append("Supertrend Bearish")
        if ema50 and cmp < ema50 and pct < -5: sell.append("Below EMA50 (-5% loss)")
        if macd_bear: sell.append("MACD Bearish Cross")
        if rsi_div["bearish_div"]: sell.append("RSI Bear Div")
        if macd_div["bearish_div"]: sell.append("MACD Bear Div")
        if "📈 Double Top" in patterns: sell.append("Double Top Rejection")
        if "🏔️ Head & Shoulders (Top)" in patterns: sell.append("H&S Bearish Reversal")
        if "🟥 Bearish Engulfing" in candles and rsi and rsi > 65: sell.append("Bearish Distribution Candle")
        if ind["vol_ratio"] > 2.5 and rsi and rsi > 65: sell.append("Volume spike at resistance")
        if trend in ("Downtrend", "Strong Downtrend") and pct < -8: sell.append(f"{trend} breakdown")
        if is_bear and pct < -5: sell.append("Bear Market override")
        # v12: momentum-fading early warning (histogram contracting + EMA flat + in profit)
        if hist_fading and ema_fading and pct > 8 and rsi and rsi > 60:
            sell.append("Momentum fading — book partial profits")
        # Trap signals
        if bull_trap:
            sell.append(f"🪤 Bull Trap (conf {bt_conf}%) — {bt_detail}")

        # ── AVERAGE / BUY Triggers ─────────────────────────────────────────────
        avg = []
        can_avg = not (trend == "Strong Downtrend" and
                       not ("📉 Double Bottom" in patterns or "🛤️ Inverse H&S (Bottom)" in patterns))
        if can_avg:
            if rsi and rsi <= 40 and pct < -5:
                if "🔨 Bullish Hammer" in candles or "🟩 Bullish Engulfing" in candles:
                    avg.append(f"Oversold Bounce Confirmed by {candles[0]}")
            if "📉 Double Bottom" in patterns and stock_rs_strong: avg.append("Double Bottom + Relative Strength")
            if "🛤️ Inverse H&S (Bottom)" in patterns: avg.append("Inverse H&S Reversal")
            if trend in ("Uptrend", "Strong Uptrend") and cmp <= ema9 * 1.015 and pct < -3:
                if "🔨 Bullish Hammer" in candles: avg.append("EMA9 Pullback + Hammer Confirmation")
            if "🚀 Vol Breakout" in patterns and pct < 0: avg.append("Reversal Vol Breakout")
            if "🚩 Bull Flag Breakout" in patterns and pct < 0: avg.append("Bull Flag Breakout")
            # v12: squeeze release entry
            if bb_squeeze and bb_breakout_up and pct < 0:
                avg.append("BB Squeeze Release ↑")
            # Bear trap = trapped sellers → reversal opportunity
            if bear_trap:
                avg.append(f"🪤 Bear Trap (conf {brt_conf}%) — {brt_detail}")

        # ── HOLD Triggers ──────────────────────────────────────────────────────
        hold = []
        if "🚀 Vol Breakout" in patterns and pct > 0: hold.append("Bullish Breakout (Hold tight)")
        if "🚩 Bull Flag Breakout" in patterns and pct > 0: hold.append("Bull Flag Continuation")
        if rsi and 45 <= rsi <= 65 and cmp > ema9 and stock_rs_strong:
            hold.append("RSI neutral, Above EMA9, Beating Index")
        elif pct > 0 and not sell: hold.append(f"In profit {pct:+.1f}%")
        elif st_bullish and trend in ("Uptrend", "Strong Uptrend") and not sell:
            hold.append("Trend & Supertrend Bullish")

        # ── Action determination — unified risk engine ─────────────────────────
        if sell:
            action      = "🔴 SELL"
            reason_base = " | ".join(sell)
            strength    = min(len(sell) * 25 + 15, 95)
            target, stop_loss, rr = _calc_risk_params(
                cmp, atr, resistance, action="SELL")
            avg_price = new_avg = new_sl = None
        elif avg:
            action      = "🟡 AVERAGE"
            reason_base = " | ".join(avg)
            strength    = min(len(avg) * 22 + 18, 88)
            avg_price   = cmp
            new_avg     = round((buy_at * qty + cmp * qty) / (qty + qty), 2)
            target, stop_loss, rr = _calc_risk_params(
                cmp, atr, resistance,
                supertrend_val=st_val, supertrend_bullish=st_bullish, action="AVERAGE")
            new_sl = stop_loss
        elif hold:
            action      = "🟢 HOLD"
            reason_base = hold[0]
            strength    = 55
            target, stop_loss, rr = _calc_risk_params(
                cmp, atr, resistance,
                supertrend_val=st_val, supertrend_bullish=st_bullish, action="HOLD")
            avg_price = new_avg = new_sl = None
        else:
            action      = "⚪ WATCH"
            reason_base = f"CMP ${cmp} | RSI {rsi if rsi else '—'} | {pct:+.1f}%"
            strength    = 30
            target, stop_loss, rr = _calc_risk_params(
                cmp, atr, resistance,
                supertrend_val=st_val, supertrend_bullish=st_bullish, action="WATCH")
            avg_price = new_avg = new_sl = None

        div_parts = []
        if rsi_div["bullish_div"]: div_parts.append("RSI Bull")
        if rsi_div["bearish_div"]: div_parts.append("RSI Bear")
        if macd_div["bullish_div"]: div_parts.append("MACD Bull")
        if macd_div["bearish_div"]: div_parts.append("MACD Bear")
        div_lbl = ", ".join(div_parts) if div_parts else "None"

        macd_lbl = "Bullish ↗" if macd_bull else ("Bearish ↘" if macd_bear else "Neutral →")
        bb_lbl   = "Lower" if bb_pos < 0.2 else ("Upper" if bb_pos > 0.8 else "Mid")
        st_lbl   = f"Bullish ${st_val}" if st_bullish else (f"Bearish ${st_val}" if st_val else "—")

        all_pats     = patterns + candles
        pat_str      = " | ".join(all_pats) if all_pats else ""
        final_reason = f"[{pat_str}] {reason_base}" if pat_str else reason_base
        if not ind.get("liquidity_ok", True):
            final_reason += " ⚠️ Low liquidity (<$1Cr/day)"

        signals.append({
            "id": tid, "stock": symbol, "sector": get_sector(symbol),
            "action": action, "reason": final_reason, "strength": strength,
            "cmp": cmp, "rsi": rsi,
            "ema20": ema9, "ema50": ema21,   # legacy display labels
            "atr": atr,
            "pct_from_buy": pct, "buy_at": buy_at, "quantity": qty,
            "target": target, "stop_loss": stop_loss,
            "avg_price": avg_price, "new_avg": new_avg, "new_sl": new_sl,
            "macd_signal": macd_lbl, "bb_position": bb_lbl,
            "trend": trend, "support": support, "resistance": resistance,
            "risk_reward": rr, "vol_ratio": ind["vol_ratio"],
            "market_regime": market["regime"], "divergence": div_lbl,
            "supertrend": st_lbl, "vwap": vwap,
            "fib_levels": {
                "23.6%": ind.get("fib_236"), "38.2%": ind.get("fib_382"),
                "50%": ind.get("fib_500"), "61.8%": ind.get("fib_618")
            },
            "limited_history": ind.get("limited_history", False),
            "bars": ind.get("bars"),
            "vcp": ind.get("vcp", False),
            "vcp_ready": ind.get("vcp_ready", False),
            "vcp_quality": ind.get("vcp_quality"),
            "rs_ratio": ind.get("rs_ratio"),
            "rs_outperforming": ind.get("rs_outperforming"),
        })
    return signals


# ─── Sector Rotation (unchanged from v11 — already 8/10) ──────────────────────
def sector_rotation(trades_df=None):
    rows = []
    idx_symbols = list(SECTOR_INDICES.values()) + ["^NSEI"]
    bulk_data   = _bulk_fetch_history(idx_symbols, period="6mo")

    nifty_df    = bulk_data.get("^NSEI")
    nifty_ret1m = nifty_ret3m = 0.0
    if nifty_df is not None and len(nifty_df) >= 21:
        nifty_ret1m = (float(nifty_df["Close"].iloc[-1]) / float(nifty_df["Close"].iloc[-21]) - 1) * 100
    if nifty_df is not None and len(nifty_df) >= 61:
        nifty_ret3m = (float(nifty_df["Close"].iloc[-1]) / float(nifty_df["Close"].iloc[-61]) - 1) * 100

    for sector, idx_sym in SECTOR_INDICES.items():
        try:
            df_idx = bulk_data.get(idx_sym)
            if df_idx is None or len(df_idx) < 21:
                continue
            close   = df_idx["Close"]
            cmp_now = float(close.iloc[-1])
            rsi     = compute_rsi(close)
            ema20   = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
            ema50   = float(close.ewm(span=50, adjust=False).mean().iloc[-1]) if len(close) >= 50 else ema20
            ret_1w  = (cmp_now / float(close.iloc[-6])  - 1) * 100 if len(close) >= 6  else 0.0
            ret_1m  = (cmp_now / float(close.iloc[-21]) - 1) * 100 if len(close) >= 21 else 0.0
            ret_3m  = (cmp_now / float(close.iloc[-61]) - 1) * 100 if len(close) >= 61 else 0.0
            rs_1m   = round(ret_1m - nifty_ret1m, 2)
            rs_3m   = round(ret_3m - nifty_ret3m, 2)
            rs_ratio    = 100 + rs_3m / 10
            rs_momentum = 100 + rs_1m / 5
            if   rs_ratio > 100 and rs_momentum > 100: rrg_quadrant = "🔥 Leading"
            elif rs_ratio > 100: rrg_quadrant = "📉 Weakening"
            elif rs_momentum > 100: rrg_quadrant = "🔄 Improving"
            else: rrg_quadrant = "❄️ Lagging"
            above_ema50 = (close > close.ewm(span=50, adjust=False).mean()).astype(int)
            streak = 0
            for val in reversed(above_ema50.values):
                if val == 1: streak += 1
                else: break
            rsi_score   = (rsi / 100) if rsi else 0.5
            rs_score    = max(-1.0, min(1.0, rs_1m / 10))
            trend_score = min(1.0, streak / 60)
            macd_score  = 1.0 if cmp_now > ema20 > ema50 else (0.5 if cmp_now > ema20 else 0.0)
            momentum_score = round(rsi_score * 0.20 + rs_score * 0.40 +
                                   trend_score * 0.20 + macd_score * 0.20, 3)
            rows.append({
                "sector": sector,
                "stocks": ", ".join(SECTOR_STOCKS.get(sector, [])[:4]) + "...",
                "cmp": round(cmp_now, 2), "rsi": rsi,
                "ret_1w": round(ret_1w, 2), "ret_1m": round(ret_1m, 2), "ret_3m": round(ret_3m, 2),
                "rs_vs_nifty_1m": rs_1m, "rs_vs_nifty_3m": rs_3m,
                "rrg_quadrant": rrg_quadrant, "trend_days": streak,
                "momentum_score": momentum_score, "macd_bullish": cmp_now > ema20,
                "count": len(SECTOR_STOCKS.get(sector, [])),
            })
        except Exception:
            continue

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values("momentum_score", ascending=False)
    df["rank"] = range(1, len(df) + 1)
    df["avg_rsi"] = df["rsi"]; df["avg_pct"] = df["ret_1m"]
    df["bullish_count"] = df["macd_bullish"].astype(int)
    df["index_chg"] = df["ret_1m"]
    return df[["rank", "sector", "stocks", "count", "avg_rsi", "avg_pct",
               "rs_vs_nifty_1m", "rs_vs_nifty_3m", "rrg_quadrant", "trend_days",
               "momentum_score", "bullish_count", "index_chg"]]


# ─── Sector Outlook ───────────────────────────────────────────────────────────
def predict_sector_outlook(sector_df):
    if sector_df.empty:
        return pd.DataFrame()
    preds = []
    for _, r in sector_df.iterrows():
        score = (r["momentum_score"] * 0.4
                 + (r.get("index_chg", 0) / 10 if r.get("index_chg") else 0) * 0.3
                 + (r["bullish_count"] / max(r["count"], 1)) * 0.3)
        if score > 0.5: outlook, conf = "🔥 Strong Bullish", 85
        elif score > 0.3: outlook, conf = "📈 Bullish", 70
        elif score > 0.1: outlook, conf = "➡️ Neutral-Bullish", 55
        elif score > -0.1: outlook, conf = "➡️ Neutral", 50
        elif score > -0.3: outlook, conf = "📉 Weak", 30
        else: outlook, conf = "🔻 Bearish", 20
        if r["avg_rsi"] and r["avg_rsi"] > 65:
            outlook = "🚀 Power Zone"; conf = min(conf + 15, 95)
        elif r["avg_rsi"] and r["avg_rsi"] < 45:
            outlook = "🩸 Bleeding — Avoid"; conf = max(conf - 20, 20)
        preds.append({"sector": r["sector"], "outlook": outlook, "confidence": conf,
                      "momentum": r["momentum_score"], "avg_rsi": r["avg_rsi"],
                      "avg_pct": r["avg_pct"], "index_chg": r.get("index_chg")})
    return pd.DataFrame(preds).sort_values("confidence", ascending=False)


# ─── Sector Stock Discovery — FIX 9: uses unified engine, action="PICK" ───────
def find_sector_picks(selected_sectors=None, max_per_sector=3):
    picks = []
    sectors = selected_sectors or list(SECTOR_STOCKS.keys())
    all_symbols = []
    for sector in sectors:
        all_symbols.extend(SECTOR_STOCKS.get(sector, []))
    all_symbols = all_symbols[:MAX_SCAN_SYMBOLS]
    bulk_data = _bulk_fetch_history(all_symbols, period="1y")

    for sector in sectors:
        spicks = []
        for symbol in SECTOR_STOCKS.get(sector, []):
            df  = bulk_data.get(symbol)
            ind = compute_indicators(symbol, period="1y", prefetched_df=df)
            if ind is None:
                continue
            if not ind.get("liquidity_ok", True):   # FIX 10: skip illiquid for new entries
                continue
            cmp, rsi = ind["cmp"], ind["rsi"]
            score, reasons = 0, []
            if rsi and 35 <= rsi <= 55: score += 15; reasons.append(f"RSI buy zone ({rsi})")
            if ind["trend"] in ("Uptrend", "Strong Uptrend"): score += 20; reasons.append(ind["trend"])
            elif ind["trend"] == "Recovery": score += 12; reasons.append("Recovery")
            if ind["macd_bullish"]: score += 15; reasons.append("MACD bullish cross")
            if ind.get("macd_hist_expanding"): score += 8; reasons.append("MACD momentum building")
            if ind.get("supertrend_bullish"): score += 10; reasons.append("Supertrend bullish")
            if ind["bb_pos"] < 0.3: score += 8; reasons.append("Lower BB bounce")
            if ind.get("bb_squeeze"): score += 8; reasons.append("BB squeeze (pre-breakout)")
            # VCP — Volatility Contraction Pattern (Minervini). Strong base = high conviction.
            if ind.get("vcp"):
                if ind.get("vcp_ready"):
                    score += 22
                    reasons.append(f"🎯 VCP pivot-ready ({ind.get('vcp_quality','')})")
                else:
                    score += 12
                    reasons.append(f"📐 VCP base ({ind.get('vcp_quality','')})")
            # Relative Strength vs Nifty — leaders get a conviction boost
            _rs = ind.get("rs_ratio")
            if _rs is not None:
                if _rs >= 1.15:
                    score += 12; reasons.append(f"💪 Strong leader (RS {_rs:.2f})")
                elif _rs >= 1.0:
                    score += 6; reasons.append(f"Outperforming Nifty (RS {_rs:.2f})")
                elif _rs < 0.85:
                    score -= 8   # laggard — reduce conviction for new longs
            if ind["vol_ratio"] > 1.3: score += 8; reasons.append(f"Vol surge ({ind['vol_ratio']:.1f}x)")
            if ind.get("bear_trap"): score += 20; reasons.append(f"🪤 Bear Trap (conf {ind.get('bear_trap_conf',0)}%)")
            if ind.get("bull_trap"): score -= 25  # avoid buying into a bull trap
            # SMC confluence — institutional structure boosts/reduces conviction
            _smc_score = ind.get("smc_score", 0)
            if _smc_score >= 35:
                score += 12; reasons.append(f"🏦 Bullish SMC ({_smc_score:+d})")
            elif _smc_score <= -35:
                score -= 12
            if ind.get("smc_zone") == "Discount":
                score += 6; reasons.append("SMC Discount zone")
            if ind.get("smc_at_bull_ob"):
                score += 8; reasons.append("At bull Order Block")
            if score < 45:
                continue

            # FIX 9: unified engine — same numbers as everywhere else
            atr = ind["atr"]
            tgt, sl, rr = _calc_risk_params(cmp, atr, ind["resistance"], action="PICK")
            if rr and rr < 1.5:
                continue

            patterns = ind.get("patterns", []); candles = ind.get("candlesticks", [])
            all_pats = patterns + candles
            pat_str  = " | ".join(all_pats) if all_pats else ""
            final_reason = (f"[{pat_str}] " + " | ".join(reasons[:4])) if pat_str else " | ".join(reasons[:4])
            spicks.append({"stock": symbol, "sector": sector, "cmp": cmp, "entry": round(cmp, 2),
                           "target": tgt, "stop_loss": sl, "risk_reward": rr, "score": score,
                           "rsi": rsi, "trend": ind["trend"], "reason": final_reason,
                           "atr": atr, "support": ind["support"], "resistance": ind["resistance"]})
        spicks.sort(key=lambda x: x["score"], reverse=True)
        picks.extend(spicks[:max_per_sector])
    picks.sort(key=lambda x: x["score"], reverse=True)
    return picks


# ─── Telegram ─────────────────────────────────────────────────────────────────
def send_telegram(bot_token, chat_id, message):
    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{bot_token}/sendMessage",
            json={"chat_id": chat_id, "text": message, "parse_mode": "HTML"}, timeout=10)
        return resp.ok
    except Exception:
        return False


def build_telegram_message(signals, sector_df, picks=None):
    now = datetime.now().strftime("%d %b %Y %H:%M")
    lines = [f"<b>📈 Swing Dashboard</b>  <i>{now}</i>\n"]
    market = get_market_regime()
    lines.append(f"🌐 <b>Market:</b> {market['regime']} | S&P {market.get('nifty_close', '—')} | RSI {market.get('nifty_rsi', '—')}")
    for s in [s for s in signals if "SELL" in s["action"]]:
        lines.append(f"🔴 <b>{s['stock']}</b> ${s['cmp']} | {s['reason']}")
    for s in [s for s in signals if "AVERAGE" in s["action"]]:
        lines.append(f"🟡 <b>{s['stock']}</b> ${s['cmp']} | Avg ${s.get('avg_price')} | New SL ${s.get('new_sl')}")
    for s in [s for s in signals if "HOLD" in s["action"]]:
        lines.append(f"🟢 <b>{s['stock']}</b> ${s['cmp']} | {s['reason']}")
    if not sector_df.empty:
        lines.append("\n🔄 <b>SECTORS</b>")
        for _, r in sector_df.head(5).iterrows():
            e = "🥇" if r["rank"] == 1 else "🥈" if r["rank"] == 2 else "📊"
            lines.append(f"  {e} <b>{r['sector']}</b> Score {r['momentum_score']:.2f}")
    if picks:
        lines.append("\n🎯 <b>BUYS</b>")
        for p in picks[:8]:
            lines.append(f"  • <b>{p['stock']}</b> ${p['cmp']} | Tgt ${p['target']} | SL ${p['stop_loss']} | R:R {p['risk_reward']}")
    lines.append("\n<i>Indicative only. Not investment advice.</i>")
    return "\n".join(lines)


# ─── Master Universe Scanner — FIX 9: unified engine + liquidity gate ─────────
def generate_market_scanner():
    all_symbols = get_scan_symbols()
    bulk_data = _bulk_fetch_history(all_symbols, period="6mo")
    results = []
    for symbol in all_symbols:
        sector = get_sector(symbol)
        df  = bulk_data.get(symbol)
        ind = compute_indicators(symbol, period="6mo", prefetched_df=df)
        if not ind:
            continue
        if not ind.get("liquidity_ok", True):   # FIX 10: visible skip for new entries
            continue
        cmp, rsi, trend = ind["cmp"], ind["rsi"], ind["trend"]
        patterns = ind.get("patterns", []); candles = ind.get("candlesticks", [])
        score = 0
        if trend in ("Uptrend", "Strong Uptrend"): score += 3
        if ind.get("supertrend_bullish"): score += 2
        if ind.get("macd_bullish"): score += 2
        if ind.get("macd_hist_expanding"): score += 1
        if ind.get("bb_squeeze"): score += 1
        if rsi and 60 <= rsi <= 75: score += 3
        if "🚀 Vol Breakout" in patterns: score += 5
        if "🚩 Bull Flag Breakout" in patterns: score += 4
        if "☕ Cup & Handle Breakout" in patterns: score += 4
        if "🟩 Bullish Engulfing" in candles: score += 2
        # VCP base adds conviction (pivot-ready more so)
        if ind.get("vcp"):
            score += 4 if ind.get("vcp_ready") else 2
        # Relative strength leadership
        _rs_sc = ind.get("rs_ratio")
        if _rs_sc is not None:
            if _rs_sc >= 1.15: score += 2
            elif _rs_sc < 0.85: score -= 2
        # Active trap is a warning — penalise
        if ind.get("bull_trap") or ind.get("bear_trap"):
            score -= 3
        if ("📈 Double Top" in patterns or "🏔️ Head & Shoulders (Top)" in patterns or
                "🟥 Bearish Engulfing" in candles):
            score -= 5
        if trend in ("Downtrend", "Strong Downtrend"): score -= 4
        if score >= 8: signal = "🔥 STRONG BUY"
        elif score >= 5: signal = "🟢 BUY SETUP"
        elif score >= 2: signal = "🟡 ACCUMULATE"
        elif score <= 0: signal = "🔴 AVOID"
        else: signal = "⚪ NEUTRAL"
        all_pats = patterns + candles
        pat_str  = " | ".join(all_pats) if all_pats else "—"

        # FIX 9: same unified engine as sector picks — identical numbers everywhere
        atr = ind["atr"]
        tgt, sl, _rr = _calc_risk_params(cmp, atr, ind["resistance"], action="PICK")

        # ── VCP / Trap / RS columns (all from the same indicator computation) ──
        if ind.get("vcp"):
            vcp_str = f"🎯 {ind.get('vcp_quality','')}" + ("▸READY" if ind.get("vcp_ready") else "")
        else:
            vcp_str = "—"
        if ind.get("bull_trap"):
            trap_str = f"🐂 Bull trap ({ind.get('bull_trap_conf',0)}%)"
        elif ind.get("bear_trap"):
            trap_str = f"🐻 Bear trap ({ind.get('bear_trap_conf',0)}%)"
        else:
            trap_str = "—"
        _rs_ratio = ind.get("rs_ratio")
        rs_str = round(float(_rs_ratio), 2) if _rs_ratio is not None else None

        results.append({
            "Generated": datetime.now().strftime("%d %b %H:%M"), "Sector": sector,
            "Stock": symbol, "CMP": float(cmp), "Entry": float(cmp),
            "Target": float(tgt), "SL": float(sl), "Support": float(ind["support"]),
            "Resist": float(ind["resistance"]), "Signal": signal, "Score": score,
            "RSI": round(float(rsi), 2) if rsi else 0.0, "Trend": trend,
            "VCP": vcp_str, "Trap": trap_str, "RS": rs_str,
            "RS_Lead": "💪" if ind.get("rs_outperforming") else "",
            "Patterns": pat_str,
            "Turnover_M": round(ind.get("avg_turnover", 0) / 1e6, 1),
            "Liquid": "✅" if ind.get("liquidity_ok", True) else "⚠️ Low",
        })
    if not results:
        return pd.DataFrame()
    return pd.DataFrame(results).sort_values(by=["Sector", "Score"], ascending=[True, False])


# ==============================================================================
# NEWS ENGINE (unchanged from v11 — already 8/10)
# ==============================================================================
def _parse_yf_news_item(item):
    if not isinstance(item, dict):
        return None, None
    content = item.get("content", {})
    if isinstance(content, dict) and content.get("title"):
        title = content.get("title", "")
        url_obj = content.get("clickThroughUrl") or content.get("canonicalUrl") or {}
        link = url_obj.get("url", "") if isinstance(url_obj, dict) else str(url_obj)
        return title, link
    title = item.get("title", "")
    link  = item.get("link", "") or item.get("url", "")
    return (title, link) if title else (None, None)


def _fetch_google_news_rss(query, max_items=2):
    try:
        q = requests.utils.quote(query)
        url = f"https://news.google.com/rss/search?q={q}+NSE+India&hl=en-IN&gl=IN&ceid=IN:en"
        resp = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        if not resp.ok:
            return []
        root = ET.fromstring(resp.content)
        items = root.findall(".//item")[:max_items]
        results = []
        for it in items:
            title_el = it.find("title")
            link_el  = it.find("link")
            if title_el is not None and title_el.text:
                title = title_el.text.strip()
                link  = link_el.text.strip() if link_el is not None and link_el.text else ""
                results.append((title, link))
        return results
    except Exception:
        return []


def fetch_portfolio_news(open_trades_df):
    if open_trades_df.empty:
        return []
    news_alerts = []
    unique_symbols = open_trades_df["stock"].unique().tolist()
    for sym in unique_symbols:
        clean_sym = sanitize_ticker(sym)
        found = False
        try:
            t     = yf.Ticker(_yahoo(clean_sym))
            items = t.get_news(count=3)
            if items:
                for item in items[:2]:
                    title, link = _parse_yf_news_item(item)
                    if title:
                        if link:
                            news_alerts.append(
                                f"📰 <b>{clean_sym}</b>: "
                                f"<a href='{link}' style='color:var(--accent);text-decoration:none'>{title}</a>")
                        else:
                            news_alerts.append(f"📰 <b>{clean_sym}</b>: {title}")
                        found = True
        except Exception:
            pass
        if not found:
            try:
                rss_items = _fetch_google_news_rss(clean_sym, max_items=1)
                for title, link in rss_items:
                    if link:
                        news_alerts.append(
                            f"📰 <b>{clean_sym}</b>: "
                            f"<a href='{link}' style='color:var(--accent);text-decoration:none'>{title}</a>")
                    else:
                        news_alerts.append(f"📰 <b>{clean_sym}</b>: {title}")
            except Exception:
                continue
    return news_alerts


# ==============================================================================
# BULL TRAP & BEAR TRAP DETECTION — v12 addition
# ==============================================================================
# Bull Trap: Price fakes a breakout above resistance → collapses back.
#            Lures buyers into a losing long. Strong SELL signal.
# Bear Trap: Price fakes a breakdown below support → snaps back up.
#            Lures sellers into a losing short. Strong BUY/AVERAGE signal.
#
# Detection uses 5-factor confluence scoring (each factor adds confidence):
#   1. False breakout/breakdown geometry (price action)
#   2. Volume quality on the fake move (weak = not a real breakout)
#   3. RSI extreme at the fake move (overbought/oversold confirmation)
#   4. Supertrend direction alignment
#   5. Candle confirmation on the reversal bar
# ==============================================================================


# ==============================================================================
# VCP — VOLATILITY CONTRACTION PATTERN  (Mark Minervini)
# ==============================================================================
# A proper VCP base, not just a Bollinger squeeze. We look for:
#   1. A prior uptrend (the stock must be a leader, not a falling knife).
#   2. A sequence of 2-4 pullbacks ("contractions") inside a base.
#   3. Each contraction SHALLOWER than the one before (e.g. -25% -> -13% -> -7%).
#   4. Volume drying up through the base (right side quieter than left).
#   5. Price coiled near the top of the base, close to the pivot (breakout point).
#   6. A "ready" flag when the final contraction is tight and price hugs pivot.
#
# Returns a dict with is_vcp, vcp_ready, contractions, pivot, quality, detail.
# ==============================================================================
def detect_vcp(close, high, low, vol, atr, lookback=80):
    out = {
        "is_vcp": False, "vcp_ready": False, "contractions": [],
        "n_contractions": 0, "pivot": None, "pivot_distance_pct": None,
        "volume_dryup": False, "base_length": 0, "quality": None, "detail": ""
    }
    n = len(close)
    if n < 30:
        return out

    c = close.values.astype(float)
    h = high.values.astype(float)
    l = low.values.astype(float)
    v = vol.values.astype(float)

    start = max(0, n - lookback)
    seg_c = c[start:]; seg_h = h[start:]; seg_l = l[start:]; seg_v = v[start:]
    m = len(seg_c)
    if m < 25:
        return out

    cmp = float(c[-1])

    # 1. Prior uptrend / leadership gate: must be basing in the top of its range
    seg_hi = float(seg_h.max()); seg_lo = float(seg_l.min())
    if seg_hi <= seg_lo:
        return out
    pos_in_range = (cmp - seg_lo) / (seg_hi - seg_lo)
    if pos_in_range < 0.55:
        out["detail"] = "Not near base highs"
        return out

    # 2. Find swing pivots (fractal highs & lows) with ATR prominence
    prom = max(atr * 0.5, cmp * 0.005)
    swing_hi_idx = []; swing_lo_idx = []
    for i in range(2, m - 2):
        if (seg_h[i] >= seg_h[i-1] and seg_h[i] >= seg_h[i-2]
                and seg_h[i] >= seg_h[i+1] and seg_h[i] >= seg_h[i+2]
                and (seg_h[i] - min(seg_l[i-2:i+3])) >= prom):
            swing_hi_idx.append(i)
        if (seg_l[i] <= seg_l[i-1] and seg_l[i] <= seg_l[i-2]
                and seg_l[i] <= seg_l[i+1] and seg_l[i] <= seg_l[i+2]
                and (max(seg_h[i-2:i+3]) - seg_l[i]) >= prom):
            swing_lo_idx.append(i)

    if len(swing_hi_idx) < 2 or len(swing_lo_idx) < 1:
        out["detail"] = "Not enough swings for a base"
        return out

    # 3. Measure contractions: each peak -> following trough drawdown
    contractions = []
    for pk in swing_hi_idx:
        later_lows = [lo for lo in swing_lo_idx if lo > pk]
        if not later_lows:
            continue
        tr = later_lows[0]
        peak_price = seg_h[pk]; trough_price = seg_l[tr]
        if peak_price > 0:
            depth = (peak_price - trough_price) / peak_price * 100.0
            if depth > 0.5:
                contractions.append(round(depth, 1))

    contractions = contractions[-4:]          # most recent base, up to 4 legs
    # Collapse consecutive near-duplicates (same peak caught by adjacent fractals)
    deduped = []
    for d in contractions:
        if not deduped or abs(deduped[-1] - d) > 1.0:
            deduped.append(d)
    contractions = deduped[-4:]
    if len(contractions) < 2:
        out["detail"] = "Fewer than 2 contractions"
        return out

    out["contractions"] = contractions
    out["n_contractions"] = len(contractions)

    # 4. Each contraction shallower than the previous (allow one noise violation)
    violations = sum(1 for i in range(1, len(contractions))
                     if contractions[i] > contractions[i-1] + 1.0)
    tightening = violations <= 1 and contractions[-1] < contractions[0]
    final_tight = contractions[-1] <= 12.0

    # 5. Volume dry-up: right third quieter than left third
    third = max(3, m // 3)
    left_vol = float(np.mean(seg_v[:third])) if third < m else float(np.mean(seg_v))
    right_vol = float(np.mean(seg_v[-third:]))
    volume_dryup = right_vol < left_vol * 0.85 if left_vol > 0 else False
    out["volume_dryup"] = volume_dryup

    # 6. Pivot = highest high of base; distance from current price
    pivot = float(seg_h.max())
    pivot_distance_pct = (pivot - cmp) / cmp * 100.0 if cmp > 0 else 999
    out["pivot"] = round(pivot, 2)
    out["pivot_distance_pct"] = round(pivot_distance_pct, 2)
    out["base_length"] = m

    is_vcp = bool(tightening and len(contractions) >= 2 and pos_in_range >= 0.55)
    out["is_vcp"] = is_vcp

    if is_vcp:
        vcp_ready = bool(final_tight and 0 <= pivot_distance_pct <= 6.0)
        out["vcp_ready"] = vcp_ready
        score = 0
        if len(contractions) >= 3: score += 1
        if tightening and violations == 0: score += 1
        if final_tight: score += 1
        if volume_dryup: score += 1
        if pivot_distance_pct <= 8: score += 1
        out["quality"] = "A+" if score >= 5 else "A" if score >= 4 else "B" if score >= 3 else "C"
        seq = " -> ".join(f"-{x}%" for x in contractions)
        ready_txt = "PIVOT-READY" if vcp_ready else f"{pivot_distance_pct:.1f}% below pivot"
        vdry = " · vol dry-up" if volume_dryup else ""
        out["detail"] = f"{len(contractions)} contractions ({seq}){vdry} · {ready_txt}"
    else:
        out["detail"] = "Contractions not tightening"

    return out


def detect_trap_signals(close, high, low, vol, vol_avg, rsi_series,
                        supertrend_bullish, resistance, support,
                        candles, atr, high52, low52, window=15,
                        market_regime=None):
    """
    Detects bull traps (false breakout above resistance) and bear traps
    (false breakdown below support).

    SCORING v4 — ATR-normalised, RSI-at-peak, distribution-aware:
    =========================================================================
    KEY UPGRADES OVER v3:
      • ATR-NORMALISED magnitude: spike/failure measured in ATR multiples,
        not raw % — so a 3% move on a low-volatility large-cap scores higher
        than a 3% move on a high-volatility small-cap (where 3% is noise).
      • RSI-AT-PEAK: RSI is read at the breakout bar (where the trap formed),
        not today's cooled-off value.
      • FAILURE-BAR VOLUME: heavy volume on the rejection = distribution,
        a much stronger trap confirmation than weak fade.
      • REPEATED REJECTION: counts how many times this level rejected price
        in the lookback — a level that rejected 3× is a real wall.
      • 52-WEEK PROXIMITY: traps near 52w highs/lows = institutional
        distribution/accumulation zones, weighted higher.
      • MARKET-REGIME CONTEXT: a bull trap in a bear market is far more
        reliable than one during a strong bull (where it's just a pullback).

    Geometry:
       [pre-period bars -60..-16] → pre_resistance / pre_support + rejection count
       [window     bars -15..-1 ] → breakout + RSI-at-peak + breakout vol
       [current    bar  0       ] → failure confirmed + failure-bar vol

    Scoring (max 100, fires at >= 55):
       Base geometry                      18
       1. Spike magnitude (ATR-norm)    0-20
       2. Failure depth (ATR-norm)      0-16
       3. Breakout volume quality       0-12
       4. Failure-bar volume (distrib)  0-12
       5. RSI-at-peak extreme           0-12
       6. Repeated rejection (wall)     0-10
       7. Reversal candle               0-10
       8. 52-week proximity             0-8
       9. Market regime alignment       0-8
      10. Supertrend (light confirm)    0-6
    =========================================================================
    """
    result = {
        "bull_trap": False, "bear_trap": False,
        "bull_trap_conf": 0, "bear_trap_conf": 0,
        "bull_trap_detail": "", "bear_trap_detail": ""
    }

    if len(close) < window + 25:
        return result

    cmp     = float(close.iloc[-1])
    pre_idx = -(window + 1)
    atr_safe = max(atr, cmp * 0.005)          # floor ATR to avoid div-by-zero

    # ── Pre-window S/R (close-based, computed BEFORE the trap window) ─────────
    if len(close) >= window + 21:
        pre_resistance = float(close.rolling(20).max().iloc[pre_idx])
        pre_support    = float(close.rolling(20).min().iloc[pre_idx])
    else:
        pre_resistance = float(close.iloc[:pre_idx].tail(20).max())
        pre_support    = float(close.iloc[:pre_idx].tail(20).min())

    tol = max(atr * 0.3, pre_resistance * 0.003)

    # Window slices
    win_close = close.iloc[-window:]
    win_vol   = vol.iloc[-window:]
    win_high  = high.iloc[-window:]
    win_low   = low.iloc[-window:]

    # RSI series aligned to window (for RSI-at-peak)
    try:
        win_rsi = rsi_series.iloc[-window:].values
    except Exception:
        win_rsi = None

    # Regime flags
    regime = (market_regime or {}).get("regime", "") if isinstance(market_regime, dict) else ""
    is_bear_mkt = regime in ("Strong Bear", "Bear", "Bear Rally")
    is_bull_mkt = regime in ("Strong Bull", "Bull")

    # ── BULL TRAP ──────────────────────────────────────────────────────────────
    breakout_idx = [i for i, c in enumerate(win_close.values)
                    if float(c) > pre_resistance + tol]
    failed_up = cmp < pre_resistance - tol

    if breakout_idx and failed_up:
        bo_closes = [float(win_close.values[i]) for i in breakout_idx]
        bo_vols   = [float(win_vol.values[i])   for i in breakout_idx]
        peak_i    = breakout_idx[int(np.argmax([float(win_close.values[i]) for i in breakout_idx]))]
        max_spike = max(bo_closes)

        # ATR-normalised magnitudes
        spike_atrs   = (max_spike - pre_resistance) / atr_safe
        failure_atrs = (pre_resistance - cmp) / atr_safe
        spike_pct    = (max_spike - pre_resistance) / max(pre_resistance, 1) * 100
        failure_pct  = (pre_resistance - cmp) / max(pre_resistance, 1) * 100

        # Hard gates: at least 0.7 ATR breakout AND 0.4 ATR failure
        if spike_atrs >= 0.7 and failure_atrs >= 0.4:
            conf = 18
            detail = [f"Failed breakout ${pre_resistance:.1f} (+{spike_pct:.1f}%)"]

            # 1. Spike magnitude (ATR-normalised)
            if   spike_atrs >= 3.0: conf += 20; detail.append(f"strong spike {spike_atrs:.1f}ATR")
            elif spike_atrs >= 1.8: conf += 13; detail.append(f"moderate spike {spike_atrs:.1f}ATR")
            elif spike_atrs >= 0.7: conf += 6

            # 2. Failure depth (ATR-normalised)
            if   failure_atrs >= 2.5: conf += 16; detail.append(f"deep rejection {failure_atrs:.1f}ATR")
            elif failure_atrs >= 1.2: conf += 9;  detail.append(f"clear rejection {failure_atrs:.1f}ATR")
            elif failure_atrs >= 0.4: conf += 4

            # 3. Breakout volume quality (weak = fake breakout)
            avg_bo_vol = sum(bo_vols) / len(bo_vols)
            bo_vr = avg_bo_vol / max(vol_avg, 1)
            if   bo_vr < 1.0: conf += 12; detail.append(f"weak breakout vol ({bo_vr:.1f}x)")
            elif bo_vr < 1.8: conf += 6;  detail.append(f"soft breakout vol ({bo_vr:.1f}x)")

            # 4. Failure-bar volume (heavy = distribution = strong)
            fail_vol_ratio = float(win_vol.values[-1]) / max(vol_avg, 1)
            if   fail_vol_ratio >= 2.0: conf += 12; detail.append(f"distribution vol ({fail_vol_ratio:.1f}x)")
            elif fail_vol_ratio >= 1.3: conf += 6;  detail.append(f"elevated sell vol ({fail_vol_ratio:.1f}x)")

            # 5. RSI AT PEAK (not today)
            if win_rsi is not None and peak_i < len(win_rsi):
                rsi_peak = win_rsi[peak_i]
                if not np.isnan(rsi_peak):
                    if   rsi_peak >= 75: conf += 12; detail.append(f"RSI {rsi_peak:.0f} at peak (extreme)")
                    elif rsi_peak >= 68: conf += 7;  detail.append(f"RSI {rsi_peak:.0f} at peak")

            # 6. Repeated rejection — how many times did price hit this zone & fail?
            zone_lo, zone_hi = pre_resistance * 0.985, pre_resistance * 1.015
            touches = int(((close.iloc[-60:] >= zone_lo) &
                           (close.iloc[-60:] <= zone_hi)).sum())
            if   touches >= 6: conf += 10; detail.append(f"strong wall ({touches} touches)")
            elif touches >= 3: conf += 5;  detail.append(f"tested level ({touches} touches)")

            # 7. Reversal candle
            reject_candles = ["🟥 Bearish Engulfing", "💫 Shooting Star",
                              "🌆 Evening Star", "🦅 Three Black Crows"]
            matched = [c for c in reject_candles if c in candles]
            if matched: conf += 10; detail.append(matched[0])

            # 8. 52-week proximity (trap near 52w high = distribution zone)
            if high52 and max_spike >= high52 * 0.97:
                conf += 8; detail.append("near 52w high (distribution)")

            # 9. Market regime: bull trap in bear market = high reliability
            if is_bear_mkt: conf += 8; detail.append(f"{regime} context")
            elif is_bull_mkt: conf -= 4    # likely just a pullback, not a trap

            # 10. Supertrend (light confirm)
            if supertrend_bullish is False:
                conf += 6; detail.append("ST bearish")

            if conf >= 55:
                result["bull_trap"]        = True
                result["bull_trap_conf"]   = int(min(max(conf, 0), 98))
                result["bull_trap_detail"] = " | ".join(detail)

    # ── BEAR TRAP ──────────────────────────────────────────────────────────────
    breakdown_idx = [i for i, c in enumerate(win_close.values)
                     if float(c) < pre_support - tol]
    failed_dn = cmp > pre_support + tol

    if breakdown_idx and failed_dn:
        bd_closes = [float(win_close.values[i]) for i in breakdown_idx]
        bd_vols   = [float(win_vol.values[i])   for i in breakdown_idx]
        trough_i  = breakdown_idx[int(np.argmin([float(win_close.values[i]) for i in breakdown_idx]))]
        max_drop  = min(bd_closes)

        drop_atrs     = (pre_support - max_drop) / atr_safe
        recovery_atrs = (cmp - pre_support) / atr_safe
        drop_pct      = (pre_support - max_drop) / max(pre_support, 1) * 100
        recovery_pct  = (cmp - pre_support) / max(pre_support, 1) * 100

        if drop_atrs >= 0.7 and recovery_atrs >= 0.4:
            conf = 18
            detail = [f"Failed breakdown ${pre_support:.1f} (-{drop_pct:.1f}%)"]

            # 1. Drop magnitude (ATR-norm)
            if   drop_atrs >= 3.0: conf += 20; detail.append(f"strong drop {drop_atrs:.1f}ATR")
            elif drop_atrs >= 1.8: conf += 13; detail.append(f"moderate drop {drop_atrs:.1f}ATR")
            elif drop_atrs >= 0.7: conf += 6

            # 2. Recovery depth (ATR-norm)
            if   recovery_atrs >= 2.5: conf += 16; detail.append(f"strong recovery {recovery_atrs:.1f}ATR")
            elif recovery_atrs >= 1.2: conf += 9;  detail.append(f"clear recovery {recovery_atrs:.1f}ATR")
            elif recovery_atrs >= 0.4: conf += 4

            # 3. Breakdown volume (weak = fake breakdown)
            avg_bd_vol = sum(bd_vols) / len(bd_vols)
            bd_vr = avg_bd_vol / max(vol_avg, 1)
            if   bd_vr < 1.0: conf += 12; detail.append(f"weak breakdown vol ({bd_vr:.1f}x)")
            elif bd_vr < 1.8: conf += 6;  detail.append(f"soft breakdown vol ({bd_vr:.1f}x)")

            # 4. Recovery-bar volume (heavy = accumulation)
            rec_vol_ratio = float(win_vol.values[-1]) / max(vol_avg, 1)
            if   rec_vol_ratio >= 2.0: conf += 12; detail.append(f"accumulation vol ({rec_vol_ratio:.1f}x)")
            elif rec_vol_ratio >= 1.3: conf += 6;  detail.append(f"elevated buy vol ({rec_vol_ratio:.1f}x)")

            # 5. RSI AT TROUGH (not today)
            if win_rsi is not None and trough_i < len(win_rsi):
                rsi_trough = win_rsi[trough_i]
                if not np.isnan(rsi_trough):
                    if   rsi_trough <= 25: conf += 12; detail.append(f"RSI {rsi_trough:.0f} at trough (extreme)")
                    elif rsi_trough <= 32: conf += 7;  detail.append(f"RSI {rsi_trough:.0f} at trough")

            # 6. Repeated rejection at support
            zone_lo, zone_hi = pre_support * 0.985, pre_support * 1.015
            touches = int(((close.iloc[-60:] >= zone_lo) &
                           (close.iloc[-60:] <= zone_hi)).sum())
            if   touches >= 6: conf += 10; detail.append(f"strong floor ({touches} touches)")
            elif touches >= 3: conf += 5;  detail.append(f"tested floor ({touches} touches)")

            # 7. Recovery candle
            recovery_candles = ["🟩 Bullish Engulfing", "🔨 Bullish Hammer",
                                "🌅 Morning Star", "🪖 Three White Soldiers",
                                "🛤️ Inverse H&S (Bottom)"]
            matched = [c for c in recovery_candles if c in candles]
            if matched: conf += 10; detail.append(matched[0])

            # 8. 52-week proximity (trap near 52w low = accumulation zone)
            if low52 and max_drop <= low52 * 1.03:
                conf += 8; detail.append("near 52w low (accumulation)")

            # 9. Market regime: bear trap in bull market = high reliability
            if is_bull_mkt: conf += 8; detail.append(f"{regime} context")
            elif is_bear_mkt: conf -= 4

            # 10. Supertrend
            if supertrend_bullish is True:
                conf += 6; detail.append("ST bullish")

            if conf >= 55:
                result["bear_trap"]        = True
                result["bear_trap_conf"]   = int(min(max(conf, 0), 98))
                result["bear_trap_detail"] = " | ".join(detail)

    return result


def scan_for_traps(min_confidence=55):
    """
    Proactively sweeps all Nifty 500 stocks for live bull trap and bear trap
    patterns. Unlike the embedded flags in generate_signals() which only cover
    your portfolio, this function scans the FULL universe and surfaces every
    trap currently forming — letting you act before you're already in a trade.

    Returns:
        {
          "bull_traps"  : list[dict]  — sorted by confidence desc
          "bear_traps"  : list[dict]  — sorted by confidence desc
          "scanned"     : int         — total symbols attempted
          "liquid"      : int         — symbols that passed liquidity gate
          "bull_count"  : int
          "bear_count"  : int
          "timestamp"   : str
        }

    Each bull_trap entry:
        stock, sector, cmp, rsi, confidence, detail,
        support, resistance, atr, stop_loss, trend,
        vol_ratio, supertrend_bullish

    Each bear_trap entry (adds trade-ready params):
        stock, sector, cmp, rsi, confidence, detail,
        support, resistance, atr, entry, target,
        stop_loss, risk_reward, trend, vol_ratio, supertrend_bullish
    """
    all_symbols = get_scan_symbols()

    # Bulk fetch — single network pass for the whole universe
    bulk_data = _bulk_fetch_history(all_symbols, period="6mo")

    bull_traps = []
    bear_traps = []
    liquid_count = 0

    for symbol in all_symbols:
        try:
            sector = get_sector(symbol)
            df  = bulk_data.get(symbol)
            ind = compute_indicators(symbol, period="6mo", prefetched_df=df)
            if not ind:
                continue
            # Liquidity gate — no point alerting on stocks you can't trade
            if not ind.get("liquidity_ok", True):
                continue
            liquid_count += 1

            cmp = ind["cmp"]
            atr = ind["atr"]

            # ── Bull Trap alert ───────────────────────────────────────────────────
            if ind.get("bull_trap") and ind.get("bull_trap_conf", 0) >= min_confidence:
                # For bull traps: target = current price (exit NOW),
                # re-entry SL shown so trader knows where to re-enter if wrong
                _, re_entry_sl, _ = _calc_risk_params(
                    cmp, atr, ind["resistance"], action="SELL"
                )
                bull_traps.append({
                    "stock":              symbol,
                    "sector":             sector,
                    "cmp":                cmp,
                    "rsi":                ind["rsi"],
                    "confidence":         ind["bull_trap_conf"],
                    "detail":             ind["bull_trap_detail"],
                    "support":            ind["support"],
                    "resistance":         ind["resistance"],
                    "atr":                atr,
                    "re_entry_sl":        re_entry_sl,
                    "trend":              ind["trend"],
                    "vol_ratio":          ind["vol_ratio"],
                    "supertrend_bullish": ind.get("supertrend_bullish"),
                    "patterns":           " | ".join(ind.get("patterns", []) + ind.get("candlesticks", [])),
                })

            # ── Bear Trap alert ───────────────────────────────────────────────────
            if ind.get("bear_trap") and ind.get("bear_trap_conf", 0) >= min_confidence:
                tgt, sl, rr = _calc_risk_params(
                    cmp, atr, ind["resistance"], action="PICK"
                )
                bear_traps.append({
                    "stock":              symbol,
                    "sector":             sector,
                    "cmp":                cmp,
                    "rsi":                ind["rsi"],
                    "confidence":         ind["bear_trap_conf"],
                    "detail":             ind["bear_trap_detail"],
                    "support":            ind["support"],
                    "resistance":         ind["resistance"],
                    "atr":                atr,
                    "entry":              round(cmp, 2),
                    "target":             tgt,
                    "stop_loss":          sl,
                    "risk_reward":        rr,
                    "trend":              ind["trend"],
                    "vol_ratio":          ind["vol_ratio"],
                    "supertrend_bullish": ind.get("supertrend_bullish"),
                    "patterns":           " | ".join(ind.get("patterns", []) + ind.get("candlesticks", [])),
                })

        except Exception:
            continue

    # Sort by confidence descending
    bull_traps.sort(key=lambda x: x["confidence"], reverse=True)
    bear_traps.sort(key=lambda x: x["confidence"], reverse=True)

    return {
        "bull_traps":  bull_traps,
        "bear_traps":  bear_traps,
        "scanned":     len(all_symbols),
        "liquid":      liquid_count,
        "bull_count":  len(bull_traps),
        "bear_count":  len(bear_traps),
        "timestamp":   datetime.now().strftime("%d %b %Y %H:%M"),
    }


# ==============================================================================
# CORPORATE ACTIONS ENGINE
# Fetches dividends, stock splits, and bonus issues for NSE stocks via yfinance.
# Cached at module level with 6-hour TTL (actions don't change intraday).
# ==============================================================================

_CORP_CACHE    = {}
_CORP_CACHE_TS = {}
_CORP_TTL      = 21600   # 6 hours


def fetch_corporate_actions(symbol):
    """
    Fetch recent + upcoming corporate actions for a single NSE stock.
    Returns:
        {
          "symbol"         : str,
          "dividends"      : list[{"date": str, "amount": float}],   # last 3
          "splits"         : list[{"date": str, "ratio": float}],     # last 2
          "upcoming_exdate": str | None,   # next ex-dividend date if known
          "last_dividend"  : float | None, # most recent dividend amount
          "last_div_date"  : str | None,
          "has_split_1y"   : bool,         # any split/bonus in last 365 days
        }
    """
    now = time.time()
    clean = sanitize_ticker(symbol)
    key   = f"corp_{clean}"

    if key in _CORP_CACHE and (now - _CORP_CACHE_TS.get(key, 0)) < _CORP_TTL:
        return _CORP_CACHE[key]

    result = {
        "symbol": clean, "dividends": [], "splits": [],
        "upcoming_exdate": None, "last_dividend": None,
        "last_div_date": None, "has_split_1y": False,
    }
    try:
        t = yf.Ticker(_yahoo(clean))

        # ── Dividends ────────────────────────────────────────────────────────
        try:
            divs = t.dividends
            if divs is not None and not divs.empty:
                recent = divs.tail(3)
                result["dividends"] = [
                    {"date": str(d.date()), "amount": round(float(v), 2)}
                    for d, v in zip(recent.index, recent.values)
                ]
                result["last_dividend"] = round(float(divs.iloc[-1]), 2)
                result["last_div_date"] = str(divs.index[-1].date())
        except Exception:
            pass

        # ── Splits / Bonus ────────────────────────────────────────────────────
        try:
            splits = t.splits
            if splits is not None and not splits.empty:
                recent = splits.tail(3)
                result["splits"] = [
                    {"date": str(d.date()), "ratio": round(float(v), 4)}
                    for d, v in zip(recent.index, recent.values)
                ]
                # Check if any split happened in the last 365 days
                cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=365)
                recent_splits = splits[splits.index >= cutoff]
                result["has_split_1y"] = not recent_splits.empty
        except Exception:
            pass

        # ── Upcoming ex-dividend date ─────────────────────────────────────────
        try:
            info = t.fast_info
            # fast_info doesn't have ex-date; try calendar
            cal = t.calendar
            if isinstance(cal, dict):
                exd = cal.get("Ex-Dividend Date") or cal.get("exDividendDate")
                if exd:
                    result["upcoming_exdate"] = str(pd.Timestamp(exd).date())
            elif isinstance(cal, pd.DataFrame) and not cal.empty:
                for col in ["Ex-Dividend Date", "exDividendDate"]:
                    if col in cal.columns:
                        val = cal[col].iloc[0]
                        if pd.notna(val):
                            result["upcoming_exdate"] = str(pd.Timestamp(val).date())
                        break
        except Exception:
            pass

    except Exception:
        pass

    _CORP_CACHE[key]    = result
    _CORP_CACHE_TS[key] = now
    return result


def fetch_bulk_corporate_actions(symbols, max_workers=8):
    """
    Batch-fetch corporate actions for a list of NSE symbols using a thread pool.
    Returns dict: {symbol: action_dict}
    Gracefully skips failures — never raises.
    """
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(fetch_corporate_actions, sym): sym
            for sym in symbols
        }
        for future in as_completed(future_map):
            sym = future_map[future]
            try:
                results[sym] = future.result()
            except Exception:
                results[sym] = {
                    "symbol": sym, "dividends": [], "splits": [],
                    "upcoming_exdate": None, "last_dividend": None,
                    "last_div_date": None, "has_split_1y": False,
                }
    return results


def scan_corporate_actions_universe(min_dividend=0.0):
    """
    Sweep the full Nifty 500 for stocks with:
      - Recent dividends (last 12 months)
      - Upcoming ex-dividend dates
      - Recent stock splits or bonus issues (last 365 days)

    Returns:
        {
          "with_upcoming_exdate" : list[dict],  # stocks with known future ex-date
          "recent_dividends"     : list[dict],  # paid dividend in last year
          "recent_splits"        : list[dict],  # split/bonus in last 365 days
          "timestamp"            : str,
          "scanned"              : int,
        }
    """
    all_symbols = get_scan_symbols()

    actions_map = fetch_bulk_corporate_actions(all_symbols)

    upcoming_exdate  = []
    recent_dividends = []
    recent_splits    = []
    today_str        = str(datetime.now().date())

    for sym, data in actions_map.items():
        sector = get_sector(sym)
        base   = {"stock": sym, "sector": sector}

        # Upcoming ex-date (future dates only)
        if data["upcoming_exdate"] and data["upcoming_exdate"] >= today_str:
            upcoming_exdate.append({
                **base,
                "ex_date":       data["upcoming_exdate"],
                "last_dividend": data["last_dividend"],
                "last_div_date": data["last_div_date"],
            })

        # Recent dividend (within last 365 days)
        if data["last_div_date"]:
            cutoff = str((datetime.now() - pd.Timedelta(days=365)).date())
            if data["last_div_date"] >= cutoff:
                if data["last_dividend"] and data["last_dividend"] >= min_dividend:
                    recent_dividends.append({
                        **base,
                        "amount":   data["last_dividend"],
                        "ex_date":  data["last_div_date"],
                    })

        # Recent splits / bonus
        if data["has_split_1y"] and data["splits"]:
            latest_split = data["splits"][-1]
            upcoming_exdate_val = data["upcoming_exdate"]
            recent_splits.append({
                **base,
                "date":  latest_split["date"],
                "ratio": latest_split["ratio"],
                "type":  "Bonus" if latest_split["ratio"] > 1 else "Split",
            })

    # Sort
    upcoming_exdate.sort(key=lambda x: x["ex_date"])
    recent_dividends.sort(key=lambda x: x["amount"], reverse=True)
    recent_splits.sort(key=lambda x: x["date"], reverse=True)

    return {
        "with_upcoming_exdate": upcoming_exdate,
        "recent_dividends":     recent_dividends,
        "recent_splits":        recent_splits,
        "scanned":              len(all_symbols),
        "timestamp":            datetime.now().strftime("%d %b %Y %H:%M"),
    }


# ==============================================================================
# SMART MONEY CONCEPTS (SMC / ICT) MODULE
# ==============================================================================
# Five institutional-footprint detectors for NSE daily charts:
#   1. detect_fvg()              — Fair Value Gaps (3-candle price inefficiency)
#   2. detect_order_blocks()     — Bullish/Bearish institutional order blocks
#   3. detect_liquidity_pools()  — Equal highs/lows (stop-loss clusters)
#   4. premium_discount_zone()   — 50% range bias (premium=sell, discount=buy)
#   5. detect_displacement()     — Strong momentum candles (institutional moves)
#
# NSE-SPECIFIC HANDLING:
#   • Circuit filters: gaps caused by 5/10/20% circuit limits are flagged,
#     not treated as genuine FVGs (they're forced moves, not inefficiencies).
#   • ATR-relative thresholds: all "significance" gates scale with volatility
#     so a small-cap and a large-cap are judged on the same relative basis.
#   • Unmitigated tracking: FVGs/OBs are flagged as still-valid only if price
#     hasn't already traded back through them.
# ==============================================================================


def detect_fvg(high, low, close, atr, lookback=30, max_zones=5):
    """
    Fair Value Gap: a 3-candle pattern where candle-1 and candle-3 do not
    overlap, leaving a price 'gap' that tends to act as a magnet.

      Bullish FVG: low[i+1] > high[i-1]   (gap below price, support zone)
      Bearish FVG: high[i+1] < low[i-1]   (gap above price, resistance zone)

    Returns dict:
      bull_fvgs / bear_fvgs : list of {top, bottom, mid, idx, mitigated, size_atr}
      nearest_bull_fvg      : closest unmitigated bullish FVG below price (or None)
      nearest_bear_fvg      : closest unmitigated bearish FVG above price (or None)
      in_bull_fvg / in_bear_fvg : bool — is current price inside an unfilled FVG
    """
    result = {
        "bull_fvgs": [], "bear_fvgs": [],
        "nearest_bull_fvg": None, "nearest_bear_fvg": None,
        "in_bull_fvg": False, "in_bear_fvg": False,
    }
    n = len(close)
    if n < 5:
        return result

    h = high.values; l = low.values; c = close.values
    cmp = float(c[-1])
    # Minimum gap size to count (filters noise): 0.25 ATR
    min_gap = max(atr * 0.25, cmp * 0.001)
    # Circuit-filter heuristic: a single-bar move > 4.5% on NSE is often a
    # circuit-driven forced move, not a genuine inefficiency. Flag those.
    start = max(2, n - lookback)

    for i in range(start, n - 1):
        # The gap-forming (middle) candle must itself be a real move, not drift.
        # Require its close-to-close move >= 0.5 ATR, else it's not displacement.
        mid_move = abs(c[i] - c[i - 1])
        if mid_move < atr * 0.5:
            continue
        # Bullish FVG: gap between high[i-1] and low[i+1]
        gap_bot = h[i - 1]
        gap_top = l[i + 1]
        if gap_top > gap_bot + min_gap:
            size = gap_top - gap_bot
            # circuit check: was the displacement candle (i) a huge single move?
            cand_move = abs(c[i] - c[i - 1]) / max(c[i - 1], 1) * 100
            is_circuit = cand_move > 4.5
            # mitigated if price has since traded back below the gap top
            mitigated = bool(np.any(l[i + 2:] <= gap_bot)) if i + 2 < n else False
            result["bull_fvgs"].append({
                "top": round(float(gap_top), 2), "bottom": round(float(gap_bot), 2),
                "mid": round(float((gap_top + gap_bot) / 2), 2),
                "idx": int(i), "mitigated": mitigated,
                "size_atr": round(size / max(atr, 0.01), 2),
                "circuit": is_circuit,
            })
        # Bearish FVG: gap between low[i-1] and high[i+1]
        gap_top2 = l[i - 1]
        gap_bot2 = h[i + 1]
        if gap_bot2 < gap_top2 - min_gap:
            size = gap_top2 - gap_bot2
            cand_move = abs(c[i] - c[i - 1]) / max(c[i - 1], 1) * 100
            is_circuit = cand_move > 4.5
            mitigated = bool(np.any(h[i + 2:] >= gap_top2)) if i + 2 < n else False
            result["bear_fvgs"].append({
                "top": round(float(gap_top2), 2), "bottom": round(float(gap_bot2), 2),
                "mid": round(float((gap_top2 + gap_bot2) / 2), 2),
                "idx": int(i), "mitigated": mitigated,
                "size_atr": round(size / max(atr, 0.01), 2),
                "circuit": is_circuit,
            })

    # Trim to most recent max_zones each
    result["bull_fvgs"] = result["bull_fvgs"][-max_zones:]
    result["bear_fvgs"] = result["bear_fvgs"][-max_zones:]

    # Nearest unmitigated, non-circuit zones
    valid_bull = [f for f in result["bull_fvgs"]
                  if not f["mitigated"] and not f["circuit"] and f["top"] < cmp]
    valid_bear = [f for f in result["bear_fvgs"]
                  if not f["mitigated"] and not f["circuit"] and f["bottom"] > cmp]
    if valid_bull:
        result["nearest_bull_fvg"] = max(valid_bull, key=lambda f: f["top"])
    if valid_bear:
        result["nearest_bear_fvg"] = min(valid_bear, key=lambda f: f["bottom"])

    # Is price currently inside an unfilled FVG?
    for f in result["bull_fvgs"]:
        if not f["mitigated"] and f["bottom"] <= cmp <= f["top"]:
            result["in_bull_fvg"] = True
    for f in result["bear_fvgs"]:
        if not f["mitigated"] and f["bottom"] <= cmp <= f["top"]:
            result["in_bear_fvg"] = True

    return result


def detect_order_blocks(open_, high, low, close, atr, lookback=40, max_blocks=4):
    """
    Order Block: the last opposing candle before a strong displacement move.
      Bullish OB: last DOWN candle before a strong UP move (institutional buying)
      Bearish OB: last UP candle before a strong DOWN move (institutional selling)

    A move qualifies as 'displacement' if it travels >= 1.5 ATR within 3 bars
    of the order-block candle.

    Returns dict:
      bull_obs / bear_obs : list of {top, bottom, mid, idx, mitigated, strength_atr}
      nearest_bull_ob     : closest unmitigated bullish OB below price (support)
      nearest_bear_ob     : closest unmitigated bearish OB above price (resistance)
      at_bull_ob / at_bear_ob : bool — price currently inside an OB zone
    """
    result = {
        "bull_obs": [], "bear_obs": [],
        "nearest_bull_ob": None, "nearest_bear_ob": None,
        "at_bull_ob": False, "at_bear_ob": False,
    }
    n = len(close)
    if n < 6:
        return result

    o = open_.values; h = high.values; l = low.values; c = close.values
    cmp = float(c[-1])
    disp_thresh = atr * 1.5
    start = max(1, n - lookback)

    for i in range(start, n - 3):
        is_down = c[i] < o[i]
        is_up   = c[i] > o[i]
        # Look at the 3 bars following the candle for displacement
        fwd_high = np.max(h[i + 1:i + 4])
        fwd_low  = np.min(l[i + 1:i + 4])
        # Impulsiveness: count directional closes in the 3 following bars.
        # A real OB is followed by an IMPULSIVE leg, not choppy sideways drift.
        fwd_up_closes   = int(np.sum(c[i + 1:i + 4] > o[i + 1:i + 4]))
        fwd_down_closes = int(np.sum(c[i + 1:i + 4] < o[i + 1:i + 4]))

        # Bullish OB: a down candle, then an IMPULSIVE up move >= 1.5 ATR
        if (is_down and (fwd_high - h[i]) >= disp_thresh
                and fwd_up_closes >= 2):
            ob_top = max(o[i], c[i]); ob_bot = l[i]
            mitigated = bool(np.any(l[i + 4:] <= ob_bot)) if i + 4 < n else False
            result["bull_obs"].append({
                "top": round(float(ob_top), 2), "bottom": round(float(ob_bot), 2),
                "mid": round(float((ob_top + ob_bot) / 2), 2),
                "idx": int(i), "mitigated": mitigated,
                "strength_atr": round((fwd_high - h[i]) / max(atr, 0.01), 2),
            })
        # Bearish OB: an up candle, then an IMPULSIVE down move >= 1.5 ATR
        if (is_up and (l[i] - fwd_low) >= disp_thresh
                and fwd_down_closes >= 2):
            ob_top = h[i]; ob_bot = min(o[i], c[i])
            mitigated = bool(np.any(h[i + 4:] >= ob_top)) if i + 4 < n else False
            result["bear_obs"].append({
                "top": round(float(ob_top), 2), "bottom": round(float(ob_bot), 2),
                "mid": round(float((ob_top + ob_bot) / 2), 2),
                "idx": int(i), "mitigated": mitigated,
                "strength_atr": round((l[i] - fwd_low) / max(atr, 0.01), 2),
            })

    result["bull_obs"] = result["bull_obs"][-max_blocks:]
    result["bear_obs"] = result["bear_obs"][-max_blocks:]

    valid_bull = [b for b in result["bull_obs"]
                  if not b["mitigated"] and b["top"] < cmp]
    valid_bear = [b for b in result["bear_obs"]
                  if not b["mitigated"] and b["bottom"] > cmp]
    if valid_bull:
        result["nearest_bull_ob"] = max(valid_bull, key=lambda b: b["top"])
    if valid_bear:
        result["nearest_bear_ob"] = min(valid_bear, key=lambda b: b["bottom"])

    for b in result["bull_obs"]:
        if not b["mitigated"] and b["bottom"] <= cmp <= b["top"]:
            result["at_bull_ob"] = True
    for b in result["bear_obs"]:
        if not b["mitigated"] and b["bottom"] <= cmp <= b["top"]:
            result["at_bear_ob"] = True

    return result


def detect_liquidity_pools(high, low, close, atr, lookback=50, tol_atr=0.15):
    """
    Liquidity Pool: clusters of near-equal highs (buy-side liquidity, where
    stops of short-sellers sit) or near-equal lows (sell-side liquidity, where
    stops of long-holders sit). Smart money targets these stop clusters.

    Two swing points are 'equal' if within tol_atr * ATR of each other.

    Returns dict:
      buyside_liquidity  : list of price levels (equal highs) above interest
      sellside_liquidity : list of price levels (equal lows)
      nearest_buyside     : closest equal-high cluster above price (or None)
      nearest_sellside    : closest equal-low cluster below price (or None)
    """
    result = {
        "buyside_liquidity": [], "sellside_liquidity": [],
        "nearest_buyside": None, "nearest_sellside": None,
    }
    n = len(close)
    if n < 10:
        return result

    h = high.values; l = low.values
    cmp = float(close.values[-1])
    tol = atr * tol_atr
    start = max(2, n - lookback)

    # Find local swing highs/lows with PROMINENCE filter.
    # A swing must stand out from its neighbours by >= 0.4 ATR to count —
    # this skips micro-swings/noise that aren't real liquidity-resting points.
    prom = atr * 0.4
    swing_highs = []
    swing_lows  = []
    for i in range(start, n - 1):
        # 3-bar fractal high that is meaningfully above both neighbours
        if (h[i] >= h[i - 1] and h[i] >= h[i + 1]
                and (h[i] - min(h[i - 1], h[i + 1])) >= prom):
            swing_highs.append((i, h[i]))
        # 3-bar fractal low that is meaningfully below both neighbours
        if (l[i] <= l[i - 1] and l[i] <= l[i + 1]
                and (max(l[i - 1], l[i + 1]) - l[i]) >= prom):
            swing_lows.append((i, l[i]))

    # Cluster equal highs (buy-side liquidity)
    used = set()
    for idx_a, (ia, pa) in enumerate(swing_highs):
        if ia in used:
            continue
        cluster = [pa]
        for ib, pb in swing_highs[idx_a + 1:]:
            if abs(pb - pa) <= tol:
                cluster.append(pb); used.add(ib)
        if len(cluster) >= 2:  # need >=2 equal highs to be a pool
            level = round(float(np.mean(cluster)), 2)
            result["buyside_liquidity"].append({"level": level, "touches": len(cluster)})

    # Cluster equal lows (sell-side liquidity)
    used = set()
    for idx_a, (ia, pa) in enumerate(swing_lows):
        if ia in used:
            continue
        cluster = [pa]
        for ib, pb in swing_lows[idx_a + 1:]:
            if abs(pb - pa) <= tol:
                cluster.append(pb); used.add(ib)
        if len(cluster) >= 2:
            level = round(float(np.mean(cluster)), 2)
            result["sellside_liquidity"].append({"level": level, "touches": len(cluster)})

    # Nearest pools relative to current price
    above = [p for p in result["buyside_liquidity"] if p["level"] > cmp]
    below = [p for p in result["sellside_liquidity"] if p["level"] < cmp]
    if above:
        result["nearest_buyside"] = min(above, key=lambda p: p["level"] - cmp)
    if below:
        result["nearest_sellside"] = min(below, key=lambda p: cmp - p["level"])

    return result


def premium_discount_zone(high, low, close, lookback=40):
    """
    Premium/Discount: divides the recent dealing range by its 50% midpoint.
      Price in PREMIUM (upper 50%)  → favour selling / caution on longs
      Price in DISCOUNT (lower 50%) → favour buying
      Equilibrium (~50%)            → neutral

    Returns dict:
      zone        : 'Premium' | 'Discount' | 'Equilibrium'
      range_high / range_low / equilibrium : float
      pct_in_range: where price sits, 0 (low) to 100 (high)
      bias        : 'Bullish' | 'Bearish' | 'Neutral'
    """
    result = {
        "zone": "Unknown", "range_high": None, "range_low": None,
        "equilibrium": None, "pct_in_range": None, "bias": "Neutral",
    }
    n = len(close)
    if n < 5:
        return result

    window = min(lookback, n)
    rng_high = float(high.values[-window:].max())
    rng_low  = float(low.values[-window:].min())
    cmp = float(close.values[-1])
    if rng_high <= rng_low:
        return result

    eq = (rng_high + rng_low) / 2
    pct = (cmp - rng_low) / (rng_high - rng_low) * 100

    if pct >= 60:
        zone, bias = "Premium", "Bearish"
    elif pct <= 40:
        zone, bias = "Discount", "Bullish"
    else:
        zone, bias = "Equilibrium", "Neutral"

    result.update({
        "zone": zone, "range_high": round(rng_high, 2), "range_low": round(rng_low, 2),
        "equilibrium": round(eq, 2), "pct_in_range": round(pct, 1), "bias": bias,
    })
    return result


def detect_displacement(open_, high, low, close, atr, lookback=10):
    """
    Displacement: a strong, large-bodied candle signalling institutional
    intent. Defined as a candle whose body >= 1.5 ATR with a close in the
    top/bottom third of its range (conviction close).

    Returns dict:
      recent_displacement : bool — any displacement in the lookback window
      direction           : 'Bullish' | 'Bearish' | None (most recent)
      bars_ago            : how many bars since the last displacement
      strength_atr        : body size of the last displacement in ATR
    """
    result = {
        "recent_displacement": False, "direction": None,
        "bars_ago": None, "strength_atr": None,
    }
    n = len(close)
    if n < 3:
        return result

    o = open_.values; h = high.values; l = low.values; c = close.values
    start = max(0, n - lookback)
    for i in range(n - 1, start - 1, -1):
        body = abs(c[i] - o[i])
        rng  = max(h[i] - l[i], 1e-9)
        # Single-candle displacement: body >= 1.5 ATR with conviction close
        if body >= atr * 1.5:
            close_pos = (c[i] - l[i]) / rng
            if c[i] > o[i] and close_pos >= 0.66:
                result.update({"recent_displacement": True, "direction": "Bullish",
                               "bars_ago": int(n - 1 - i),
                               "strength_atr": round(body / max(atr, 0.01), 2)})
                return result
            if c[i] < o[i] and close_pos <= 0.34:
                result.update({"recent_displacement": True, "direction": "Bearish",
                               "bars_ago": int(n - 1 - i),
                               "strength_atr": round(body / max(atr, 0.01), 2)})
                return result
        # Multi-candle leg: 2-3 consecutive same-direction candles covering
        # >= 2 ATR total also counts as institutional displacement.
        if i >= 2:
            up3   = all(c[j] > o[j] for j in (i-2, i-1, i))
            dn3   = all(c[j] < o[j] for j in (i-2, i-1, i))
            leg   = abs(c[i] - o[i-2])
            if (up3 or dn3) and leg >= atr * 2.0:
                result.update({
                    "recent_displacement": True,
                    "direction": "Bullish" if up3 else "Bearish",
                    "bars_ago": int(n - 1 - i),
                    "strength_atr": round(leg / max(atr, 0.01), 2)})
                return result
    return result


def build_smc_setup(cmp, atr, fvg, obs, liq, pdz, disp, score):
    """
    Converts raw SMC structure into a single actionable trade setup with
    concrete Entry, Target, and Stop-Loss — all derived from real structural
    levels, not arbitrary ATR multiples.

    LOGIC
    ─────────────────────────────────────────────────────────────────────────
    BUY setup (needs bullish confluence):
      • Trigger: price in/near a bull Order Block OR bull FVG, in Discount or
        Equilibrium, ideally with recent bullish displacement.
      • Entry : top of the bull OB (or FVG mid) — where demand sits.
      • SL    : just below the OB/FVG bottom (structure invalidation).
      • Target: nearest buy-side liquidity above (equal highs = stop cluster
        price is drawn toward); fallback to range high / 2R.

    SELL setup (mirror, bearish confluence):
      • Entry : bottom of bear OB (or FVG mid).
      • SL    : just above the OB/FVG top.
      • Target: nearest sell-side liquidity below; fallback range low / 2R.

    Returns dict:
      smc_action      : 'BUY' | 'SELL' | 'WAIT'
      smc_entry/target/sl : float | None
      smc_rr          : float | None  (reward:risk)
      smc_setup_quality : 'A+' | 'A' | 'B' | None  (confluence grade)
      smc_setup_reason  : str  (human explanation)
    ─────────────────────────────────────────────────────────────────────────
    """
    out = {
        "smc_action": "WAIT", "smc_entry": None, "smc_target": None,
        "smc_sl": None, "smc_rr": None, "smc_setup_quality": None,
        "smc_setup_reason": "No high-confluence SMC setup right now.",
    }

    sl_buffer = max(atr * 0.25, cmp * 0.002)   # small buffer beyond structure

    # ── BUY SETUP ──────────────────────────────────────────────────────────────
    bull_zone = obs.get("nearest_bull_ob") or (
        {"top": obs["at_bull_ob"]} if False else None)
    # Prefer an OB the price is AT or just above; else nearest bull FVG
    buy_zone = None; buy_src = None
    if obs.get("at_bull_ob") and obs.get("nearest_bull_ob"):
        buy_zone = obs["nearest_bull_ob"]; buy_src = "Order Block"
    elif obs.get("nearest_bull_ob"):
        buy_zone = obs["nearest_bull_ob"]; buy_src = "Order Block"
    elif fvg.get("in_bull_fvg") and fvg.get("nearest_bull_fvg"):
        buy_zone = fvg["nearest_bull_fvg"]; buy_src = "Fair Value Gap"
    elif fvg.get("nearest_bull_fvg"):
        buy_zone = fvg["nearest_bull_fvg"]; buy_src = "Fair Value Gap"

    # ── SELL SETUP ─────────────────────────────────────────────────────────────
    sell_zone = None; sell_src = None
    if obs.get("at_bear_ob") and obs.get("nearest_bear_ob"):
        sell_zone = obs["nearest_bear_ob"]; sell_src = "Order Block"
    elif obs.get("nearest_bear_ob"):
        sell_zone = obs["nearest_bear_ob"]; sell_src = "Order Block"
    elif fvg.get("in_bear_fvg") and fvg.get("nearest_bear_fvg"):
        sell_zone = fvg["nearest_bear_fvg"]; sell_src = "Fair Value Gap"
    elif fvg.get("nearest_bear_fvg"):
        sell_zone = fvg["nearest_bear_fvg"]; sell_src = "Fair Value Gap"

    zone_pct = pdz.get("pct_in_range")
    disp_dir = disp.get("direction")
    disp_recent = disp.get("bars_ago") is not None and disp.get("bars_ago") <= 4

    # Proximity gate: only emit a setup if CMP is within a workable distance of
    # the entry zone (<= 4 ATR away). Prevents suggesting an entry far from the
    # current price, which is the #1 source of confusing/unusable signals.
    max_dist = atr * 4.0
    def _zone_near(zone):
        if not zone:
            return False
        z_top = float(zone.get("top", 0)); z_bot = float(zone.get("bottom", 0))
        z_mid = (z_top + z_bot) / 2
        return abs(cmp - z_mid) <= max_dist

    buy_near  = _zone_near(buy_zone)
    sell_near = _zone_near(sell_zone)

    # ── Decide direction by confluence score + zone availability + proximity ───
    want_buy  = score >= 25 and buy_zone is not None and buy_near
    want_sell = score <= -25 and sell_zone is not None and sell_near

    # If both or neither, pick the stronger-aligned one
    if want_buy and not want_sell:
        decision = "BUY"
    elif want_sell and not want_buy:
        decision = "SELL"
    elif want_buy and want_sell:
        decision = "BUY" if score > 0 else "SELL"
    else:
        decision = "WAIT"

    # ── Build BUY ──────────────────────────────────────────────────────────────
    if decision == "BUY" and buy_zone:
        entry = float(buy_zone["top"])           # enter at demand top
        zbot  = float(buy_zone["bottom"])
        sl    = round(zbot - sl_buffer, 2)
        # Target = nearest buy-side liquidity above, else range high, else 2R
        tgt = None
        nbs = liq.get("nearest_buyside")
        if nbs and nbs["level"] > entry:
            tgt = float(nbs["level"])
        elif pdz.get("range_high") and pdz["range_high"] > entry:
            tgt = float(pdz["range_high"])
        risk = max(entry - sl, 0.01)
        if tgt is None or tgt <= entry:
            tgt = round(entry + risk * 2, 2)     # fallback 2R
        rr = round((tgt - entry) / risk, 2)

        # Quality grade by confluence (+ RR magnitude bonus)
        confl = 0
        if pdz.get("bias") == "Bullish": confl += 1
        if buy_src == "Order Block":     confl += 1
        if disp_dir == "Bullish" and disp_recent: confl += 1
        if nbs:                          confl += 1
        if rr >= 3.0:                    confl += 1   # strong RR adds conviction
        quality = "A+" if confl >= 4 else "A" if confl == 3 else "B"

        reasons = [f"Buy at {buy_src} ${entry}"]
        if zone_pct is not None: reasons.append(f"{pdz.get('zone')} zone ({zone_pct}%)")
        if disp_dir == "Bullish" and disp_recent: reasons.append("bullish displacement")
        if nbs: reasons.append(f"target buy-side liq ${tgt}")

        # Only surface if RR is worthwhile
        if rr >= 1.3:
            out.update({
                "smc_action": "BUY", "smc_entry": round(entry, 2),
                "smc_target": round(tgt, 2), "smc_sl": sl, "smc_rr": rr,
                "smc_setup_quality": quality,
                "smc_setup_reason": " · ".join(reasons),
            })

    # ── Build SELL ─────────────────────────────────────────────────────────────
    elif decision == "SELL" and sell_zone:
        entry = float(sell_zone["bottom"])       # enter at supply bottom
        ztop  = float(sell_zone["top"])
        sl    = round(ztop + sl_buffer, 2)
        tgt = None
        nss = liq.get("nearest_sellside")
        if nss and nss["level"] < entry:
            tgt = float(nss["level"])
        elif pdz.get("range_low") and pdz["range_low"] < entry:
            tgt = float(pdz["range_low"])
        risk = max(sl - entry, 0.01)
        if tgt is None or tgt >= entry:
            tgt = round(entry - risk * 2, 2)
        rr = round((entry - tgt) / risk, 2)

        confl = 0
        if pdz.get("bias") == "Bearish": confl += 1
        if sell_src == "Order Block":    confl += 1
        if disp_dir == "Bearish" and disp_recent: confl += 1
        if nss:                          confl += 1
        if rr >= 3.0:                    confl += 1
        quality = "A+" if confl >= 4 else "A" if confl == 3 else "B"

        reasons = [f"Sell at {sell_src} ${entry}"]
        if zone_pct is not None: reasons.append(f"{pdz.get('zone')} zone ({zone_pct}%)")
        if disp_dir == "Bearish" and disp_recent: reasons.append("bearish displacement")
        if nss: reasons.append(f"target sell-side liq ${tgt}")

        if rr >= 1.3:
            out.update({
                "smc_action": "SELL", "smc_entry": round(entry, 2),
                "smc_target": round(tgt, 2), "smc_sl": sl, "smc_rr": rr,
                "smc_setup_quality": quality,
                "smc_setup_reason": " · ".join(reasons),
            })

    return out


def compute_smc(open_, high, low, close, vol, atr):
    """
    Master SMC aggregator — runs all five detectors and returns a flat dict
    of the most actionable signals plus a combined SMC bias score.

    smc_bias: -100 (strong bearish confluence) to +100 (strong bullish).
    """
    fvg  = detect_fvg(high, low, close, atr)
    obs  = detect_order_blocks(open_, high, low, close, atr)
    liq  = detect_liquidity_pools(high, low, close, atr)
    pdz  = premium_discount_zone(high, low, close)
    disp = detect_displacement(open_, high, low, close, atr)

    # Combined bias score from confluence
    score = 0
    if pdz["bias"] == "Bullish": score += 20
    elif pdz["bias"] == "Bearish": score -= 20
    if fvg["in_bull_fvg"]: score += 15
    if fvg["in_bear_fvg"]: score -= 15
    if obs["at_bull_ob"]: score += 20
    if obs["at_bear_ob"]: score -= 20
    if disp["direction"] == "Bullish" and disp["bars_ago"] is not None and disp["bars_ago"] <= 3:
        score += 15
    elif disp["direction"] == "Bearish" and disp["bars_ago"] is not None and disp["bars_ago"] <= 3:
        score -= 15
    # Nearest unfilled FVG acting as a pull — but DON'T double-count if price
    # is already inside an FVG (that's already scored above).
    if fvg["nearest_bull_fvg"] and not fvg["in_bull_fvg"]: score += 8
    if fvg["nearest_bear_fvg"] and not fvg["in_bear_fvg"]: score -= 8
    score = max(-100, min(100, score))

    if score >= 35:   smc_label = "Bullish SMC"
    elif score <= -35: smc_label = "Bearish SMC"
    else:             smc_label = "Neutral SMC"

    # Build the actionable trade setup from the structure
    cmp = float(close.iloc[-1]) if len(close) else 0.0
    setup = build_smc_setup(cmp, atr, fvg, obs, liq, pdz, disp, score)

    return {
        # Premium/Discount
        "smc_zone": pdz["zone"], "smc_zone_pct": pdz["pct_in_range"],
        "smc_bias": pdz["bias"], "smc_equilibrium": pdz["equilibrium"],
        "smc_range_high": pdz["range_high"], "smc_range_low": pdz["range_low"],
        # FVG
        "smc_in_bull_fvg": fvg["in_bull_fvg"], "smc_in_bear_fvg": fvg["in_bear_fvg"],
        "smc_nearest_bull_fvg": fvg["nearest_bull_fvg"],
        "smc_nearest_bear_fvg": fvg["nearest_bear_fvg"],
        "smc_bull_fvg_count": len([f for f in fvg["bull_fvgs"] if not f["mitigated"]]),
        "smc_bear_fvg_count": len([f for f in fvg["bear_fvgs"] if not f["mitigated"]]),
        # Order Blocks
        "smc_at_bull_ob": obs["at_bull_ob"], "smc_at_bear_ob": obs["at_bear_ob"],
        "smc_nearest_bull_ob": obs["nearest_bull_ob"],
        "smc_nearest_bear_ob": obs["nearest_bear_ob"],
        # Liquidity
        "smc_nearest_buyside": liq["nearest_buyside"],
        "smc_nearest_sellside": liq["nearest_sellside"],
        # Displacement
        "smc_displacement": disp["direction"],
        "smc_displacement_bars_ago": disp["bars_ago"],
        # Combined
        "smc_score": score, "smc_label": smc_label,
        # Actionable trade setup
        "smc_action": setup["smc_action"],
        "smc_entry": setup["smc_entry"],
        "smc_target": setup["smc_target"],
        "smc_sl": setup["smc_sl"],
        "smc_rr": setup["smc_rr"],
        "smc_setup_quality": setup["smc_setup_quality"],
        "smc_setup_reason": setup["smc_setup_reason"],
    }


# ==============================================================================
# SMC SETUP SCANNER — sweeps universe for actionable SMC trade setups
# ==============================================================================
def scan_for_smc_setups(min_quality="B", action_filter="All"):
    """
    Sweeps the full universe for stocks with an actionable SMC trade setup
    (BUY or SELL) with concrete Entry/Target/SL/RR.

    Args:
        min_quality   : 'A+', 'A', or 'B' — minimum setup grade to include
        action_filter : 'All', 'BUY', or 'SELL'

    Returns:
        {
          "buy_setups"  : list[dict] sorted by quality then RR,
          "sell_setups" : list[dict],
          "scanned": int, "liquid": int,
          "buy_count": int, "sell_count": int,
          "timestamp": str,
        }
    Each setup: stock, sector, cmp, action, entry, target, stop_loss,
                risk_reward, quality, reason, smc_score, zone
    """
    quality_rank = {"A+": 3, "A": 2, "B": 1}
    min_rank = quality_rank.get(min_quality, 1)

    all_symbols = get_scan_symbols()

    bulk = _bulk_fetch_history(all_symbols, period="6mo")

    buy_setups, sell_setups = [], []
    liquid = 0

    for symbol in all_symbols:
        df  = bulk.get(symbol)
        ind = compute_indicators(symbol, period="6mo", prefetched_df=df)
        if not ind:
            continue
        if not ind.get("liquidity_ok", True):
            continue
        liquid += 1

        action  = ind.get("smc_action", "WAIT")
        quality = ind.get("smc_setup_quality")
        if action == "WAIT" or not quality:
            continue
        if quality_rank.get(quality, 0) < min_rank:
            continue
        if action_filter != "All" and action != action_filter:
            continue

        entry = {
            "stock": symbol, "sector": get_sector(symbol),
            "cmp": ind.get("cmp"), "action": action,
            "entry": ind.get("smc_entry"), "target": ind.get("smc_target"),
            "stop_loss": ind.get("smc_sl"), "risk_reward": ind.get("smc_rr"),
            "quality": quality, "reason": ind.get("smc_setup_reason", ""),
            "smc_score": ind.get("smc_score", 0), "zone": ind.get("smc_zone", ""),
        }
        if action == "BUY":
            buy_setups.append(entry)
        else:
            sell_setups.append(entry)

    def _sort_key(s):
        return (quality_rank.get(s["quality"], 0), s.get("risk_reward") or 0)
    buy_setups.sort(key=_sort_key, reverse=True)
    sell_setups.sort(key=_sort_key, reverse=True)

    return {
        "buy_setups": buy_setups, "sell_setups": sell_setups,
        "scanned": len(all_symbols), "liquid": liquid,
        "buy_count": len(buy_setups), "sell_count": len(sell_setups),
        "timestamp": datetime.now().strftime("%d %b %Y %H:%M"),
    }

def scan_for_vcp(min_quality="B", ready_only=False):
    """
    Sweeps the universe for stocks forming a Volatility Contraction Pattern.

    Args:
        min_quality : 'A+', 'A', 'B', or 'C' — minimum base grade to include
        ready_only  : if True, only return pivot-ready bases (coiled at breakout)

    Returns:
        {
          "vcp_setups": list[dict] sorted by quality then pivot proximity,
          "scanned": int, "liquid": int, "count": int, "ready_count": int,
          "timestamp": str,
        }
    Each setup: stock, sector, cmp, pivot, pivot_distance_pct, quality,
                contractions, vcp_ready, detail, entry, target, stop_loss
    """
    quality_rank = {"A+": 4, "A": 3, "B": 2, "C": 1}
    min_rank = quality_rank.get(min_quality, 2)

    all_symbols = get_scan_symbols()

    bulk = _bulk_fetch_history(all_symbols, period="6mo")

    vcp_setups = []
    liquid = 0
    ready_count = 0

    for symbol in all_symbols:
        df  = bulk.get(symbol)
        ind = compute_indicators(symbol, period="6mo", prefetched_df=df)
        if not ind:
            continue
        if not ind.get("liquidity_ok", True):
            continue
        liquid += 1

        if not ind.get("vcp"):
            continue
        quality = ind.get("vcp_quality")
        if quality_rank.get(quality, 0) < min_rank:
            continue
        is_ready = ind.get("vcp_ready", False)
        if ready_only and not is_ready:
            continue
        if is_ready:
            ready_count += 1

        cmp   = ind.get("cmp")
        pivot = ind.get("vcp_pivot")
        atr   = ind.get("atr", 0)
        # Entry just above pivot; stop below last contraction low (~1.5 ATR);
        # target a measured move (pivot + the first/biggest contraction depth).
        entry = round(pivot * 1.002, 2) if pivot else cmp
        stop  = round(entry - 1.5 * atr, 2) if atr else (round(entry * 0.93, 2) if entry else None)
        contractions = ind.get("vcp_contractions", [])
        move_pct = (max(contractions) / 100.0) if contractions else 0.10
        target = round(entry * (1 + max(move_pct, 0.08)), 2) if entry else None
        rr = round((target - entry) / (entry - stop), 2) if (entry and stop and entry > stop) else None

        vcp_setups.append({
            "stock": symbol, "sector": get_sector(symbol), "cmp": cmp,
            "pivot": pivot, "pivot_distance_pct": ind.get("vcp_pivot_dist"),
            "quality": quality, "contractions": contractions,
            "vcp_ready": is_ready, "detail": ind.get("vcp_detail", ""),
            "entry": entry, "target": target, "stop_loss": stop, "risk_reward": rr,
        })

    def _sort_key(s):
        # ready first, then quality, then closest to pivot
        return (1 if s["vcp_ready"] else 0,
                quality_rank.get(s["quality"], 0),
                -(s.get("pivot_distance_pct") or 999))
    vcp_setups.sort(key=_sort_key, reverse=True)

    return {
        "vcp_setups": vcp_setups,
        "scanned": len(all_symbols), "liquid": liquid,
        "count": len(vcp_setups), "ready_count": ready_count,
        "timestamp": datetime.now().strftime("%d %b %Y %H:%M"),
    }

def scan_relative_strength(top_n=None, min_rating=0):
    """
    Rank the entire universe by Relative Strength versus Nifty.

    Computes each liquid stock's RS ratio, then converts to a 1-99 percentile
    RS Rating (IBD-style): 99 = strongest leader, 1 = weakest laggard.

    Args:
        top_n      : if set, return only the top N leaders
        min_rating : only include stocks with RS Rating >= this (0-99)

    Returns:
        {
          "leaders": list[dict] sorted by RS rating desc,
          "scanned": int, "liquid": int, "count": int,
          "nifty_returns": dict, "timestamp": str,
        }
    Each row: stock, sector, cmp, rs_ratio, rs_rating, ret_21d, ret_63d,
              ret_252d, nifty_21d, outperforming, trend
    """
    bench = _get_nifty_benchmark()
    if bench is None:
        return {"leaders": [], "scanned": 0, "liquid": 0, "count": 0,
                "nifty_returns": {}, "timestamp": datetime.now().strftime("%d %b %Y %H:%M"),
                "error": "Could not fetch Nifty benchmark data"}

    all_symbols = get_scan_symbols()

    bulk = _bulk_fetch_history(all_symbols, period="1y")

    rows = []
    ratios = []
    liquid = 0
    for symbol in all_symbols:
        df  = bulk.get(symbol)
        ind = compute_indicators(symbol, period="1y", prefetched_df=df)
        if not ind:
            continue
        if not ind.get("liquidity_ok", True):
            continue
        liquid += 1
        rs_ratio = ind.get("rs_ratio")
        if rs_ratio is None:
            continue
        ratios.append(rs_ratio)
        periods = ind.get("rs_periods") or {}
        rows.append({
            "stock": symbol, "sector": get_sector(symbol), "cmp": ind.get("cmp"),
            "rs_ratio": rs_ratio, "outperforming": ind.get("rs_outperforming"),
            "trend": ind.get("trend"),
            "ret_21d": periods.get("21", {}).get("stock"),
            "ret_63d": periods.get("63", {}).get("stock"),
            "ret_252d": periods.get("252", {}).get("stock"),
            "nifty_21d": periods.get("21", {}).get("nifty"),
            "vcp": ind.get("vcp", False), "vcp_ready": ind.get("vcp_ready", False),
        })

    # Assign 1-99 percentile rating across all measured stocks
    for r in rows:
        r["rs_rating"] = _rs_ratio_to_rating(r["rs_ratio"], ratios)

    # Filter + sort
    rows = [r for r in rows if (r["rs_rating"] or 0) >= min_rating]
    rows.sort(key=lambda r: r["rs_rating"] or 0, reverse=True)
    if top_n:
        rows = rows[:top_n]

    return {
        "leaders": rows,
        "scanned": len(all_symbols), "liquid": liquid, "count": len(rows),
        "nifty_returns": bench,
        "timestamp": datetime.now().strftime("%d %b %Y %H:%M"),
    }
