#!/usr/bin/env python3
"""
build_us_universe.py — Liquidity-filter the US universe.
================================================================================
Reads  us_universe.csv  (stocks + ETFs, ~4,800 symbols)
Fetches recent price + average dollar volume from Yahoo Finance, keeps only
LIQUID, tradeable names, and writes  us_universe_liquid.csv  (Symbol,Name,Sector).

The Streamlit app prefers us_universe_liquid.csv if it exists, else falls back
to us_universe.csv. Run this:
  • locally:        python build_us_universe.py
  • or on a cron:   GitHub Actions nightly (see .github/workflows/refresh.yml)

This is intentionally a SEPARATE offline step. Scanning ~4,800 live tickers on
every Streamlit page load would time out and hit Yahoo rate limits, so we bake
the liquid list once and ship the small CSV.
================================================================================
"""
import os
import sys
import time
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    sys.exit("yfinance not installed.  Run:  pip install yfinance")

# ── Liquidity thresholds — tune to taste ──────────────────────────────────────
MIN_PRICE        = 5.0          # drop sub-$5 names (penny stocks, illiquid)
MIN_AVG_DOLLAR_VOL = 5_000_000  # min 20-day average dollar volume ($5M/day)
LOOKBACK_DAYS    = "1mo"        # window for the average-volume calc
BATCH            = 200          # tickers per yf.download batch
PAUSE            = 1.0          # seconds between batches (politeness)

_BASE = os.path.dirname(os.path.abspath(__file__))
SRC   = os.path.join(_BASE, "us_universe.csv")
OUT   = os.path.join(_BASE, "us_universe_liquid.csv")


def _avg_dollar_volume(df):
    """20d average of (Close * Volume) for a single-ticker OHLCV frame."""
    if df is None or df.empty or "Close" not in df or "Volume" not in df:
        return None, None
    closes = df["Close"].dropna()
    vols   = df["Volume"].dropna()
    if closes.empty or vols.empty:
        return None, None
    last_price = float(closes.iloc[-1])
    dollar_vol = (closes * vols).tail(20).mean()
    return last_price, float(dollar_vol)


def main():
    uni = pd.read_csv(SRC)
    uni.columns = [c.strip() for c in uni.columns]
    symbols = uni["Symbol"].astype(str).str.strip().str.upper().tolist()
    meta = {r["Symbol"].strip().upper(): (r["Name"], r["Sector"])
            for _, r in uni.iterrows()}

    print(f"Universe: {len(symbols):,} symbols")
    print(f"Filter:   price >= ${MIN_PRICE}  AND  avg$vol >= ${MIN_AVG_DOLLAR_VOL:,.0f}/day\n")

    kept = []
    for i in range(0, len(symbols), BATCH):
        batch = symbols[i:i + BATCH]
        try:
            data = yf.download(batch, period=LOOKBACK_DAYS, interval="1d",
                               auto_adjust=False, progress=False,
                               threads=True, group_by="ticker")
        except Exception as e:
            print(f"  batch {i}-{i+len(batch)} failed: {e}")
            time.sleep(PAUSE)
            continue

        for sym in batch:
            try:
                df = data[sym] if sym in data else None
            except Exception:
                df = None
            price, dvol = _avg_dollar_volume(df)
            if price is None or dvol is None:
                continue
            if price >= MIN_PRICE and dvol >= MIN_AVG_DOLLAR_VOL:
                name, sector = meta.get(sym, (sym, "Others"))
                kept.append({"Symbol": sym, "Name": name, "Sector": sector})

        done = min(i + BATCH, len(symbols))
        print(f"  scanned {done:,}/{len(symbols):,}  →  kept {len(kept):,} so far")
        time.sleep(PAUSE)

    out = pd.DataFrame(kept).drop_duplicates(subset=["Symbol"])
    out.to_csv(OUT, index=False)
    print(f"\n✅ Wrote {len(out):,} liquid symbols → {OUT}")
    print(out["Sector"].value_counts().to_string())


if __name__ == "__main__":
    main()
