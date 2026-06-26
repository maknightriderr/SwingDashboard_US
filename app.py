"""
Swing Trading Portfolio Dashboard v14
Fixes vs v13:
  - Premium theme pack: 6 institutional themes (was 3)
  - theme_css upgraded: animated title underline, card shimmer/lift, live-pulse
    badge, focus-glow inputs, P&L row rails, tabular numerals
  - Tab 9 scorecard updated to signals.py v12 (avg 8.4, every component >= 8)
  - signals.py v12 already deployed: unified risk engine, Wilder ATR/RSI,
    numpy Supertrend, 20-day VWAP, swing-peak Fibonacci, MACD histogram
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import sqlite3
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
import hashlib

from signals import (
    generate_signals, sector_rotation, predict_sector_outlook,
    find_sector_picks, send_telegram, build_telegram_message,
    get_sector, get_market_regime, generate_market_scanner,
    SECTOR_MAP, _bulk_fetch_history, compute_indicators,
    fetch_portfolio_news, UNIVERSE_SOURCES, UNIVERSE_TOTAL,
    debug_universe_load, MAX_SCAN_SYMBOLS
)

# New functions added in signals.py v12+ — imported separately so the app
# degrades gracefully if an older signals.py is deployed.
try:
    from signals import scan_for_traps as _scan_for_traps
    scan_for_traps = _scan_for_traps
except ImportError:
    scan_for_traps = None

try:
    from signals import scan_for_smc_setups as _scan_for_smc_setups
    scan_for_smc_setups = _scan_for_smc_setups
except ImportError:
    scan_for_smc_setups = None

try:
    from signals import scan_for_vcp as _scan_for_vcp
    scan_for_vcp = _scan_for_vcp
except ImportError:
    scan_for_vcp = None

try:
    from signals import scan_relative_strength as _scan_rs
    scan_relative_strength = _scan_rs
except ImportError:
    scan_relative_strength = None

try:
    from signals import (
        fetch_corporate_actions,
        fetch_bulk_corporate_actions,
        scan_corporate_actions_universe,
    )
except ImportError:
    fetch_corporate_actions        = None
    fetch_bulk_corporate_actions   = None
    scan_corporate_actions_universe = None

_TRAP_SCANNER_AVAILABLE = scan_for_traps is not None
_CORP_ACTIONS_AVAILABLE = fetch_corporate_actions is not None
_SMC_SCANNER_AVAILABLE  = scan_for_smc_setups is not None

# ── Mutual Fund & ETF module (fully separate from stock/signals logic) ─────────
try:
    import funds as _funds
    _FUNDS_AVAILABLE = True
except Exception:
    _funds = None
    _FUNDS_AVAILABLE = False

# ── Performance: @st.cache_data wrappers ──────────────────────────────────────
# market_regime is global (same for all users) — safe to cache across sessions.
# TTL 600s = 10 min. This renders the header banner in <100ms on reruns.
@st.cache_data(ttl=600, show_spinner=False)
def _cached_market_regime():
    return get_market_regime()


def _get_market_regime_safe():
    """Wrapper that avoids caching an empty (failed) regime result.
    If indices came back empty, clear the cache so the next rerun retries."""
    m = _cached_market_regime()
    if not m or not m.get("indices"):
        # Empty/failed — drop the cached empty so next call re-fetches fresh
        try:
            _cached_market_regime.clear()
        except Exception:
            pass
        # Try one direct (uncached) fetch right now
        try:
            m2 = get_market_regime()
            if m2 and m2.get("indices"):
                return m2
        except Exception:
            pass
    return m or {"regime": "Unknown", "indices": {}, "confidence": "—"}

# Price cache: 5-min TTL so KPI cards don't block on every sidebar interaction.
@st.cache_data(ttl=120, show_spinner=False)
def _cached_prices(symbols_tuple):
    """Fetch ACCURATE live prices for a tuple of symbols.

    Accuracy notes:
      • Primary source is fast_info.last_price — the real-time last traded price.
      • History fallback uses auto_adjust=FALSE so we get the ACTUAL close, not a
        dividend/split back-adjusted value (auto_adjust=True was making CMP wrong
        for stocks that recently went ex-dividend or split).
      • 2-min cache (was 5) so prices are fresher during market hours.
    Never raises — missing prices just stay absent."""
    import yfinance as _yf
    import pandas as _pd
    prices = {}
    if not symbols_tuple:
        return prices

    sym_map = {}   # ticker_ns → original sym
    for sym in symbols_tuple:
        clean = str(sym).upper().strip()
        for sfx in [".NS", ".BO", ".NSE", ".BSE"]:
            if clean.endswith(sfx):
                clean = clean[:-len(sfx)]
        sym_map[clean] = sym

    def _extract_fast_price(t):
        """Real-time last traded price from fast_info (most accurate source)."""
        try:
            fi = t.fast_info
        except Exception:
            return None
        for key in ("last_price", "lastPrice", "regularMarketPrice"):
            for getter in (lambda: fi.get(key) if hasattr(fi, "get") else None,
                           lambda: getattr(fi, key, None),
                           lambda: fi[key] if hasattr(fi, "__getitem__") else None):
                try:
                    v = getter()
                    if v is not None and not _pd.isna(v) and float(v) > 0:
                        return float(v)
                except Exception:
                    pass
        return None

    # ── METHOD 1: per-ticker fast_info (LIVE last traded price = most accurate) ─
    for tk, sym in sym_map.items():
        base = tk
        for sfx in [""]:
            try:
                t = _yf.Ticker(base + sfx)
                v = _extract_fast_price(t)
                if v is not None and v > 0:
                    prices[sym] = round(v, 2)
                    break
            except Exception:
                continue

    # ── METHOD 2: batch download for any the live method missed ────────────────
    # auto_adjust=False → ACTUAL close price (not back-adjusted for div/splits)
    missing = [tk for tk, sym in sym_map.items() if sym not in prices]
    if missing:
        try:
            data = _yf.download(missing, period="5d", interval="1d",
                                auto_adjust=False, progress=False,
                                group_by="ticker", threads=True)
            if data is not None and not data.empty:
                for tk in missing:
                    try:
                        if len(missing) == 1:
                            close_ser = data["Close"] if "Close" in data.columns else None
                        else:
                            close_ser = (data[tk]["Close"]
                                         if tk in data.columns.get_level_values(0) else None)
                        if close_ser is not None:
                            valid = close_ser.dropna()
                            if not valid.empty:
                                prices[sym_map[tk]] = round(float(valid.iloc[-1]), 2)
                    except Exception:
                        continue
        except Exception:
            pass

    # ── METHOD 3: per-ticker history fallback (auto_adjust=False) ──────────────
    for tk, sym in sym_map.items():
        if sym in prices:
            continue
        base = tk
        for sfx in [""]:
            try:
                t = _yf.Ticker(base + sfx)
                h = t.history(period="5d", interval="1d", auto_adjust=False)
                if h is not None and not h.empty and "Close" in h.columns:
                    valid = h["Close"].dropna()
                    if not valid.empty:
                        prices[sym] = round(float(valid.iloc[-1]), 2)
                        break
            except Exception:
                continue
    return prices

# ── Auto-refresh ───────────────────────────────────────────────────────────────
REFRESH_SEC = 300
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=REFRESH_SEC * 1000, key="dashboard_autorefresh")
except ImportError:
    pass

st.set_page_config(
    page_title="Swing Dashboard", page_icon="📈",
    layout="wide", initial_sidebar_state="expanded"
)

# ── Auth helpers ───────────────────────────────────────────────────────────────
def make_hash(password):
    return hashlib.sha256(str.encode(password + "swing_salt_99")).hexdigest()

def verify_hash(password, hashed_pw):
    return make_hash(password) == hashed_pw

# ── Database ───────────────────────────────────────────────────────────────────
# Uses Neon DB (serverless Postgres). Neon is reachable from Streamlit Cloud and
# has no schema-resolution quirks. Connection params are passed as keyword
# are passed as explicit keyword args to psycopg2 — no URL string parsing.
# ==============================================================================

DB = "trades_us.db"   # SQLite store for the US app (its OWN DB, separate from NSE)

# ── DATABASE CONNECTION ───────────────────────────────────────────────────────
# Postgres (e.g. Neon) is OPTIONAL. Provide credentials via Streamlit secrets or
# environment variables; if none are found the app uses local SQLite (DB above).
# In Streamlit Cloud → Settings → Secrets, add a [postgres] section:
#   [postgres]
#   host="..."  port=5432  dbname="..."  user="..."  password="..."  sslmode="require"
# IMPORTANT: use a SEPARATE database from the NSE app so US/NSE trades never mix.
def _load_pg_params():
    import os
    try:
        sec = st.secrets.get("postgres", None)
    except Exception:
        sec = None
    src = dict(sec) if sec else {}
    host = src.get("host") or os.environ.get("PG_HOST")
    if not host:
        return None
    return dict(
        host            = host,
        port            = int(src.get("port") or os.environ.get("PG_PORT", 5432)),
        dbname          = src.get("dbname") or os.environ.get("PG_DBNAME", "neondb"),
        user            = src.get("user") or os.environ.get("PG_USER"),
        password        = src.get("password") or os.environ.get("PG_PASSWORD"),
        sslmode         = src.get("sslmode") or os.environ.get("PG_SSLMODE", "require"),
        connect_timeout = 15,
    )

_PG_PARAMS = _load_pg_params()
_USE_PG = _PG_PARAMS is not None

try:
    import psycopg2
    import psycopg2.extras
except ImportError:
    _USE_PG = False


def _pg_conn():
    """Open a Neon Postgres connection using explicit keyword params.
    Neon is serverless — the compute may be asleep and take a few seconds to
    wake on the first connection, so we retry several times with backoff."""
    last_err = None
    for _attempt in range(4):
        try:
            conn = psycopg2.connect(**_PG_PARAMS)
            conn.autocommit = False
            return conn
        except Exception as e:
            last_err = e
            time.sleep(1.0 * (_attempt + 1))   # 1s, 2s, 3s backoff for cold start
    raise last_err

def _q(sql):
    """Translate SQLite SQL → Postgres '%s' placeholders and 'INSERT OR REPLACE'.
    Schema qualification is handled directly in db() below."""
    if not _USE_PG:
        return sql
    s = sql.replace("?", "%s")
    s = s.replace("INSERT OR REPLACE INTO", "INSERT INTO")
    return s


_PG_SCHEMA_PREFIX = "public."
_PG_TABLES = ("users", "trades", "portfolio_history", "tg_config", "watchlist", "price_alerts", "trade_journal")


def _pg_qualify(sql):
    """No-op now. We rely on 'SET search_path TO public' (which Neon fully
    supports) rather than hardcoding public. prefixes. Kept as a function so
    callers don't need to change. Returns sql unchanged."""
    return sql


def db(sql, params=(), fetch=False):
    if _USE_PG:
        conn = _pg_conn()
        cur  = conn.cursor()
        # Neon fully supports search_path (unlike Supabase pooler). Set it so
        # bare table names resolve to public.* — this is the standard approach.
        try:
            cur.execute("SET search_path TO public")
        except Exception:
            pass
        # Also qualify explicitly as belt-and-suspenders.
        pg_sql = _pg_qualify(_q(sql))
        try:
            cur.execute(pg_sql, params)
            conn.commit()
        except Exception as e:
            conn.rollback()
            cur.close(); conn.close()
            # Surface the real SQL + error so the cause is visible, not redacted.
            raise RuntimeError(f"DB error on [{pg_sql}]: {type(e).__name__}: {e}") from e
        result = cur.fetchall() if fetch else None
        cur.close(); conn.close()
        return result
    else:
        conn = sqlite3.connect(DB)
        cur = conn.execute(sql, params)
        conn.commit()
        result = cur.fetchall() if fetch else None
        conn.close()
        return result

def init_db():
    if _USE_PG:
        conn = _pg_conn(); cur = conn.cursor()
        try:
            cur.execute("SET search_path TO public")
            conn.commit()
        except Exception:
            conn.rollback()
        # Create each table in its own transaction so one failure doesn't
        # abort the others (a failed statement poisons the whole transaction
        # in Postgres until rollback).
        table_ddls = [
            """CREATE TABLE IF NOT EXISTS users(
                id SERIAL PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL)""",
            """CREATE TABLE IF NOT EXISTS trades(
                id SERIAL PRIMARY KEY, user_id INTEGER, stock TEXT NOT NULL,
                quantity REAL NOT NULL, buy_at REAL NOT NULL, sell_at REAL,
                status TEXT DEFAULT 'Open',
                added_date TEXT DEFAULT to_char(CURRENT_DATE,'YYYY-MM-DD'),
                closed_date TEXT)""",
            """CREATE TABLE IF NOT EXISTS portfolio_history(
                id SERIAL PRIMARY KEY, user_id INTEGER, snapshot_date TEXT,
                total_invested REAL, current_value REAL)""",
            """CREATE TABLE IF NOT EXISTS tg_config(
                user_id INTEGER PRIMARY KEY, bot_token TEXT, chat_id TEXT)""",
            """CREATE TABLE IF NOT EXISTS watchlist(
                id SERIAL PRIMARY KEY, user_id INTEGER, stock TEXT NOT NULL,
                target_price REAL, notes TEXT,
                added_date TEXT DEFAULT to_char(CURRENT_DATE,'YYYY-MM-DD'))""",
            """CREATE TABLE IF NOT EXISTS price_alerts(
                id SERIAL PRIMARY KEY, user_id INTEGER, stock TEXT NOT NULL,
                condition TEXT NOT NULL, target_price REAL NOT NULL,
                status TEXT DEFAULT 'Active', note TEXT,
                created_date TEXT DEFAULT to_char(CURRENT_DATE,'YYYY-MM-DD'),
                triggered_date TEXT)""",
            """CREATE TABLE IF NOT EXISTS trade_journal(
                id SERIAL PRIMARY KEY, user_id INTEGER, stock TEXT NOT NULL,
                trade_date TEXT, direction TEXT, entry_price REAL, exit_price REAL,
                setup TEXT, rationale TEXT, emotion TEXT, outcome TEXT,
                lesson TEXT, rating INTEGER,
                created_date TEXT DEFAULT to_char(CURRENT_DATE,'YYYY-MM-DD'))""",
        ]
        for ddl in table_ddls:
            try:
                cur.execute(ddl)
                conn.commit()
            except Exception:
                conn.rollback()
        cur.close(); conn.close()
    else:
        c = sqlite3.connect(DB)
        c.execute("""CREATE TABLE IF NOT EXISTS users(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL)""")
        c.execute("""CREATE TABLE IF NOT EXISTS trades(
            id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, stock TEXT NOT NULL,
            quantity REAL NOT NULL, buy_at REAL NOT NULL, sell_at REAL,
            status TEXT DEFAULT 'Open', added_date TEXT DEFAULT(date('now')),
            closed_date TEXT)""")
        c.execute("""CREATE TABLE IF NOT EXISTS portfolio_history(
            id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, snapshot_date TEXT,
            total_invested REAL, current_value REAL)""")
        c.execute("""CREATE TABLE IF NOT EXISTS tg_config(
            user_id INTEGER PRIMARY KEY, bot_token TEXT, chat_id TEXT)""")
        c.execute("""CREATE TABLE IF NOT EXISTS watchlist(
            id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, stock TEXT NOT NULL,
            target_price REAL, notes TEXT, added_date TEXT DEFAULT(date('now')))""")
        c.execute("""CREATE TABLE IF NOT EXISTS price_alerts(
            id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, stock TEXT NOT NULL,
            condition TEXT NOT NULL, target_price REAL NOT NULL,
            status TEXT DEFAULT 'Active', note TEXT,
            created_date TEXT DEFAULT(date('now')), triggered_date TEXT)""")
        c.execute("""CREATE TABLE IF NOT EXISTS trade_journal(
            id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, stock TEXT NOT NULL,
            trade_date TEXT, direction TEXT, entry_price REAL, exit_price REAL,
            setup TEXT, rationale TEXT, emotion TEXT, outcome TEXT,
            lesson TEXT, rating INTEGER, created_date TEXT DEFAULT(date('now')))""")
        c.commit(); c.close()

def register_user(username, password):
    try:
        db("INSERT INTO users(username, password_hash) VALUES(?,?)",
           (username.lower(), make_hash(password)))
        return True
    except Exception:
        return False

def login_user(username, password):
    user = db("SELECT id, password_hash FROM users WHERE username=?",
              (username.lower(),), fetch=True)
    if user and verify_hash(password, user[0][1]):
        return user[0][0]
    return None

def get_trades(user_id):
    if _USE_PG:
        conn = _pg_conn()
        try:
            cur = conn.cursor(); cur.execute("SET search_path TO public"); cur.close()
        except Exception:
            pass
        df = pd.read_sql_query(
            "SELECT * FROM trades WHERE user_id=%s ORDER BY id DESC",
            conn, params=(user_id,))
        conn.close()
        return df
    conn = sqlite3.connect(DB)
    df = pd.read_sql_query(
        "SELECT * FROM trades WHERE user_id=? ORDER BY id DESC",
        conn, params=(user_id,))
    conn.close()
    return df

def get_history(user_id):
    if _USE_PG:
        conn = _pg_conn()
        try:
            cur = conn.cursor(); cur.execute("SET search_path TO public"); cur.close()
        except Exception:
            pass
        df = pd.read_sql_query(
            "SELECT * FROM portfolio_history WHERE user_id=%s ORDER BY snapshot_date",
            conn, params=(user_id,))
        conn.close()
        return df
    conn = sqlite3.connect(DB)
    df = pd.read_sql_query(
        "SELECT * FROM portfolio_history WHERE user_id=? ORDER BY snapshot_date",
        conn, params=(user_id,))
    conn.close()
    return df

def get_watchlist(user_id):
    if _USE_PG:
        conn = _pg_conn()
        try:
            cur = conn.cursor(); cur.execute("SET search_path TO public"); cur.close()
        except Exception:
            pass
        df = pd.read_sql_query(
            "SELECT * FROM watchlist WHERE user_id=%s ORDER BY id DESC",
            conn, params=(user_id,))
        conn.close()
        return df
    conn = sqlite3.connect(DB)
    df = pd.read_sql_query(
        "SELECT * FROM watchlist WHERE user_id=? ORDER BY id DESC",
        conn, params=(user_id,))
    conn.close()
    return df

def get_tg_config(user_id):
    rows = db("SELECT bot_token,chat_id FROM tg_config WHERE user_id=?",
              (user_id,), fetch=True)
    return rows[0] if rows else ("", "")

def save_tg_config(user_id, token, chat):
    if _USE_PG:
        db("INSERT INTO tg_config(user_id,bot_token,chat_id) VALUES(?,?,?) "
           "ON CONFLICT(user_id) DO UPDATE SET bot_token=EXCLUDED.bot_token, "
           "chat_id=EXCLUDED.chat_id",
           (user_id, token, chat))
    else:
        db("INSERT OR REPLACE INTO tg_config(user_id,bot_token,chat_id) VALUES(?,?,?)",
           (user_id, token, chat))

def add_trade(user_id, stock, qty, buy, sell=None):
    status = "Closed" if sell else "Open"
    closed = datetime.now().strftime("%Y-%m-%d") if sell else None
    db("INSERT INTO trades(user_id,stock,quantity,buy_at,sell_at,status,closed_date) VALUES(?,?,?,?,?,?,?)",
       (user_id, stock.upper().strip(), qty, buy, sell, status, closed))

def update_trade(tid, user_id, stock, qty, buy, sell, status):
    closed = datetime.now().strftime("%Y-%m-%d") if status == "Closed" else None
    db("UPDATE trades SET stock=?,quantity=?,buy_at=?,sell_at=?,status=?,closed_date=? WHERE id=? AND user_id=?",
       (stock.upper().strip(), qty, buy, sell, status, closed, tid, user_id))

def delete_trade(tid, user_id):
    db("DELETE FROM trades WHERE id=? AND user_id=?", (tid, user_id))

def close_trade(tid, user_id, sell):
    db("UPDATE trades SET sell_at=?,status='Closed',closed_date=? WHERE id=? AND user_id=?",
       (sell, datetime.now().strftime("%Y-%m-%d"), tid, user_id))

def save_snapshot(user_id, invested, value):
    """Save a daily portfolio snapshot. Non-critical — if it fails (e.g. DB
    hiccup), it must NOT crash the dashboard, so errors are swallowed."""
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        db("DELETE FROM portfolio_history WHERE snapshot_date=? AND user_id=?", (today, user_id))
        db("INSERT INTO portfolio_history(user_id,snapshot_date,total_invested,current_value) VALUES(?,?,?,?)",
           (user_id, today, invested, value))
    except Exception:
        pass   # snapshot is non-essential; never block the dashboard on it

def add_watchlist(user_id, stock, target=None, notes=""):
    db("INSERT INTO watchlist(user_id,stock,target_price,notes) VALUES(?,?,?,?)",
       (user_id, stock.upper().strip(), target, notes))

def delete_watchlist_item(wid, user_id):
    db("DELETE FROM watchlist WHERE id=? AND user_id=?", (wid, user_id))

# ── Price Alert helpers ────────────────────────────────────────────────────────
def add_price_alert(user_id, stock, condition, target_price, note=""):
    """condition is 'above' or 'below'."""
    db("INSERT INTO price_alerts(user_id,stock,condition,target_price,note) "
       "VALUES(?,?,?,?,?)",
       (user_id, stock.upper().strip(), condition, float(target_price), note))

def get_price_alerts(user_id, status=None):
    if status:
        rows = db("SELECT id,stock,condition,target_price,status,note,"
                  "created_date,triggered_date FROM price_alerts "
                  "WHERE user_id=? AND status=? ORDER BY id DESC",
                  (user_id, status), fetch=True)
    else:
        rows = db("SELECT id,stock,condition,target_price,status,note,"
                  "created_date,triggered_date FROM price_alerts "
                  "WHERE user_id=? ORDER BY id DESC", (user_id,), fetch=True)
    return rows or []

def delete_price_alert(aid, user_id):
    db("DELETE FROM price_alerts WHERE id=? AND user_id=?", (aid, user_id))

def trigger_price_alert(aid, user_id):
    today = datetime.now().strftime("%Y-%m-%d")
    db("UPDATE price_alerts SET status='Triggered',triggered_date=? "
       "WHERE id=? AND user_id=?", (today, aid, user_id))

# ── Trade Journal helpers ──────────────────────────────────────────────────────
def add_journal_entry(user_id, stock, trade_date, direction, entry, exit_p,
                      setup, rationale, emotion, outcome, lesson, rating):
    db("INSERT INTO trade_journal(user_id,stock,trade_date,direction,entry_price,"
       "exit_price,setup,rationale,emotion,outcome,lesson,rating) "
       "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
       (user_id, stock.upper().strip(), trade_date, direction, entry, exit_p,
        setup, rationale, emotion, outcome, lesson, rating))

def get_journal_entries(user_id):
    rows = db("SELECT id,stock,trade_date,direction,entry_price,exit_price,setup,"
              "rationale,emotion,outcome,lesson,rating,created_date "
              "FROM trade_journal WHERE user_id=? ORDER BY id DESC",
              (user_id,), fetch=True)
    return rows or []

def delete_journal_entry(jid, user_id):
    db("DELETE FROM trade_journal WHERE id=? AND user_id=?", (jid, user_id))

# ── Session & Cookie Init ──────────────────────────────────────────────────────
from streamlit_cookies_controller import CookieController
controller = CookieController(key='app_cookies')

# Initialise DB and record connection status for the sidebar badge
_DB_STATUS = "sqlite"
_DB_ERROR = None
try:
    init_db()
    _DB_STATUS = "postgres" if _USE_PG else "sqlite"
except Exception as _db_e:
    _DB_ERROR = str(_db_e)
    # If Postgres was configured but failed, fall back to SQLite so the app
    # still loads (data won't persist, but the user isn't locked out).
    if _USE_PG:
        _USE_PG = False
        _DB_STATUS = "sqlite_fallback"
        try:
            init_db()
        except Exception:
            pass

# Ensure session state variables exist
for k, v in [("user_id", None), ("username", None), ("edit_id", None), ("close_id", None), ("del_id", None),
             ("last_refresh", None), ("last_auto_scan", 0.0), ("last_slow_scan", 0.0),
             ("_trade_hash", -1), ("sort_col", "stock"), ("sort_asc", False),
             ("signals_cache", None), ("sector_cache", None), ("picks_cache", None),
             ("outlook_cache", None), ("scanner_cache", None), ("trap_scan_cache", None),
             ("corp_actions_cache", None), ("selected_scanner_sector", "All Sectors"),
             ("custom_stocks_input", ""), ("active_page", "portfolio"),
             ("smc_scan_cache", None), ("vcp_scan_cache", None), ("rs_scan_cache", None),
             ("etf_scan_cache", None), ("mf_search_results", []),
             ("mf_selected", None), ("mf_compare_list", []),
             ("_earnings_cache", None), ("_ipo_watch", []),
             ("first_render_done", False), ("_kickoff_scan", False),
             ("_scan_stage", "done"), ("_deep_stage", "sector"),
             ("_deep_running", False), ("_manual_deep_request", False),
             ("_manual_fast_request", False),
             ("_run_deep_now", False), ("_deep_progress", "done"),
             ("fast_interval_sec", 300), ("deep_interval_sec", 900),
             ("auto_fast", True), ("auto_deep", True),
             ("filter_status", "All"),
             ("filter_pnl", "All"), ("search", ""), ("theme", "Obsidian & Gold (Institutional)")]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── Auth Gate ──────────────────────────────────────────────────────────────────
if st.session_state.user_id is None:
    cookies = controller.getAll()

    if cookies is None:
        st.info("Loading secure tunnel...")
        time.sleep(0.5)
        st.rerun()

    if cookies and cookies.get("swing_user_id"):
        try:
            cookie_uid = int(cookies.get("swing_user_id"))
            st.session_state.user_id = cookie_uid
            user_row = db("SELECT username FROM users WHERE id=?"
                        
            if user_row:
                st.session_state.username = user_row[0][0]
                st.session_state.first_render_done = False  # defer scans
                st.rerun()
        except Exception:
            pass

    st.markdown(
        "<h1 style='text-align:center;margin-top:5rem'>🔐 Quantitative Swing Dashboard</h1>",
        unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align:center;color:gray'>Secure Multi-Tenant Gateway</p>",
        unsafe_allow_html=True)

    _, auth_col, _ = st.columns([1, 1.5, 1])
    with auth_col:
        tab_login, tab_signup = st.tabs(["Login", "Create Account"])

        with tab_login:
            with st.form("login_form"):
                l_user = st.text_input("Username")
                l_pass = st.text_input("Password", type="password")
                if st.form_submit_button("Access Terminal", width="stretch"):
                    uid = login_user(l_user, l_pass)
                    if uid:
                        st.session_state.user_id = uid
                        st.session_state.username = l_user
                        st.session_state.first_render_done = False  # defer scans
                        controller.set("swing_user_id", str(uid), max_age=604800)
                        st.success("Authenticated. Booting Engine...")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("❌ Invalid Username or Password")

        with tab_signup:
            with st.form("signup_form"):
                s_user = st.text_input("New Username")
                s_pass = st.text_input("New Password", type="password")
                if st.form_submit_button("Register Account", width="stretch"):
                    if len(s_user) < 3 or len(s_pass) < 4:
                        st.error("Username > 3 chars and Password > 4 chars required.")
                    else:
                        if register_user(s_user, s_pass):
                            st.success(f"✅ Account {s_user} registered!")
                        else:
                            st.error("❌ Username already exists.")
    st.stop()

# ==============================================================================
# MAIN APPLICATION (Only runs if Authenticated)
# ==============================================================================

# User ID strictly injected into all DB calls below
UID = st.session_state.user_id

# ── HARDCODED SIDEBAR TOGGLE ───────────────────────────────────────────────────
# Streamlit's native expand control is unreliable across versions and themes.
# This injects an always-present floating button that finds and clicks the real
# (possibly hidden) sidebar toggle via JS. Works on desktop AND mobile.
components.html("""
<script>
(function() {
    const doc = window.parent.document;
    if (doc.getElementById('hard-sidebar-toggle')) return;  // inject once

    const btn = doc.createElement('button');
    btn.id = 'hard-sidebar-toggle';
    btn.innerHTML = '☰';
    btn.title = 'Toggle sidebar';
    btn.style.cssText = [
        'position:fixed', 'top:10px', 'left:10px', 'z-index:2147483647',
        'width:42px', 'height:42px', 'border-radius:10px', 'border:none',
        'background:#d4af37', 'color:#000', 'font-size:20px', 'font-weight:800',
        'cursor:pointer', 'box-shadow:0 3px 14px rgba(0,0,0,.55)',
        'display:flex', 'align-items:center', 'justify-content:center',
        'transition:transform .15s ease'
    ].join(';');
    btn.onmouseover = () => btn.style.transform = 'scale(1.08)';
    btn.onmouseout  = () => btn.style.transform = 'scale(1)';

    btn.onclick = function() {
        // Try every known selector for the sidebar toggle, in priority order.
        const selectors = [
            '[data-testid="stSidebarCollapsedControl"] button',
            '[data-testid="stSidebarCollapsedControl"]',
            '[data-testid="collapsedControl"] button',
            '[data-testid="collapsedControl"]',
            '[data-testid="stSidebarCollapseButton"] button',
            '[data-testid="stSidebarCollapseButton"]',
            '[aria-label="Open sidebar"]',
            '[aria-label="Close sidebar"]'
        ];
        for (const sel of selectors) {
            const el = doc.querySelector(sel);
            if (el) { el.click(); return; }
        }
        // Last resort: toggle the sidebar width directly
        const sb = doc.querySelector('[data-testid="stSidebar"]');
        if (sb) {
            const hidden = sb.getAttribute('aria-expanded') === 'false'
                        || sb.style.transform.includes('-');
            sb.style.transform = hidden ? 'translateX(0)' : 'translateX(-100%)';
            sb.style.visibility = 'visible';
        }
    };
    doc.body.appendChild(btn);
})();
</script>
""", height=0)

THEMES = {
    # ── 1. The flagship: obsidian black + champagne gold, private-bank feel ──
    "Obsidian & Gold (Institutional)": {
        "bg": "#050608", "card": "rgba(13, 14, 18, 0.85)", "input": "#15171c",
        "border": "rgba(212, 175, 55, 0.18)",
        "text": "#fdfdfd", "muted": "#8e8e93",
        "green": "#10b981", "red": "#ef4444", "yellow": "#d4af37",
        "blue": "#3b82f6", "accent": "#d4af37", "card2": "#121419",
        "gradient": "linear-gradient(145deg, rgba(212,175,55,0.04) 0%, rgba(13,14,18,0.95) 35%, #050608 100%)",
        "glow": "rgba(212, 175, 55, 0.25)",
        "bg_fx": "radial-gradient(ellipse 80% 50% at 50% -20%, rgba(212,175,55,0.06), transparent)",
    },
    # ── 2. Bloomberg-terminal energy: near-black + signal orange ─────────────
    "Terminal Amber (Bloomberg)": {
        "bg": "#0a0a0a", "card": "rgba(18, 16, 12, 0.9)", "input": "#1a1813",
        "border": "rgba(255, 153, 0, 0.16)",
        "text": "#f5f0e8", "muted": "#9a917f",
        "green": "#33d17a", "red": "#ff5547", "yellow": "#ff9900",
        "blue": "#4da6ff", "accent": "#ff9900", "card2": "#161410",
        "gradient": "linear-gradient(160deg, rgba(255,153,0,0.05) 0%, rgba(18,16,12,0.95) 40%, #0a0a0a 100%)",
        "glow": "rgba(255, 153, 0, 0.22)",
        "bg_fx": "radial-gradient(ellipse 70% 45% at 80% -10%, rgba(255,153,0,0.05), transparent)",
    },
    # ── 3. Deep sapphire glassmorphism — frosted panels over midnight blue ───
    "Deep Sapphire (Glass)": {
        "bg": "#020617", "card": "rgba(15, 23, 42, 0.55)", "input": "#1e293b",
        "border": "rgba(56, 189, 248, 0.14)",
        "text": "#f8fafc", "muted": "#94a3b8",
        "green": "#10b981", "red": "#f43f5e", "yellow": "#f59e0b",
        "blue": "#0ea5e9", "accent": "#38bdf8", "card2": "rgba(30, 41, 59, 0.45)",
        "gradient": "linear-gradient(135deg, rgba(56,189,248,0.06) 0%, rgba(15,23,42,0.85) 45%, rgba(2,6,23,0.95) 100%)",
        "glow": "rgba(56, 189, 248, 0.25)",
        "bg_fx": "radial-gradient(ellipse 60% 40% at 20% -10%, rgba(56,189,248,0.08), transparent), radial-gradient(ellipse 50% 35% at 90% 10%, rgba(99,102,241,0.05), transparent)",
    },
    # ── 4. Emerald quant desk — money green on graphite ──────────────────────
    "Emerald Quant (Hedge Fund)": {
        "bg": "#060a08", "card": "rgba(11, 18, 14, 0.88)", "input": "#13201a",
        "border": "rgba(16, 185, 129, 0.16)",
        "text": "#f0fdf6", "muted": "#7e9a8c",
        "green": "#10b981", "red": "#f43f5e", "yellow": "#eab308",
        "blue": "#22d3ee", "accent": "#34d399", "card2": "#0e1812",
        "gradient": "linear-gradient(150deg, rgba(16,185,129,0.05) 0%, rgba(11,18,14,0.94) 40%, #060a08 100%)",
        "glow": "rgba(52, 211, 153, 0.22)",
        "bg_fx": "radial-gradient(ellipse 75% 50% at 50% -15%, rgba(16,185,129,0.06), transparent)",
    },
    # ── 5. Royal violet — premium fintech (Zerodha-dark x Stripe) ────────────
    "Royal Violet (Fintech)": {
        "bg": "#08060f", "card": "rgba(18, 13, 30, 0.88)", "input": "#1c1430",
        "border": "rgba(167, 139, 250, 0.16)",
        "text": "#faf8ff", "muted": "#9b8fc0",
        "green": "#34d399", "red": "#fb7185", "yellow": "#fbbf24",
        "blue": "#818cf8", "accent": "#a78bfa", "card2": "#150f26",
        "gradient": "linear-gradient(140deg, rgba(167,139,250,0.06) 0%, rgba(18,13,30,0.94) 40%, #08060f 100%)",
        "glow": "rgba(167, 139, 250, 0.25)",
        "bg_fx": "radial-gradient(ellipse 65% 45% at 30% -10%, rgba(167,139,250,0.07), transparent), radial-gradient(ellipse 50% 35% at 85% 5%, rgba(244,114,182,0.04), transparent)",
    },
    # ── 6. Carbon matrix — monochrome quant, teal data accents ───────────────
    "Carbon Matrix (Quant)": {
        "bg": "#09090b", "card": "rgba(18, 18, 20, 0.92)", "input": "#18181b",
        "border": "rgba(255, 255, 255, 0.07)",
        "text": "#fafafa", "muted": "#a1a1aa",
        "green": "#22c55e", "red": "#ff3366", "yellow": "#f59e0b",
        "blue": "#06b6d4", "accent": "#14b8a6", "card2": "#141416",
        "gradient": "linear-gradient(180deg, rgba(20,184,166,0.04) 0%, rgba(18,18,20,0.96) 35%, #09090b 100%)",
        "glow": "rgba(20, 184, 166, 0.20)",
        "bg_fx": "radial-gradient(ellipse 70% 45% at 50% -15%, rgba(20,184,166,0.05), transparent)",
    },
}

# --- Fail-safe to prevent KeyErrors when a saved theme name no longer exists ---
if st.session_state.theme not in THEMES:
    st.session_state.theme = "Obsidian & Gold (Institutional)"

def theme_css(t):
    glow  = t.get("glow", "rgba(255,255,255,0.1)")
    bg_fx = t.get("bg_fx", "none")
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600;9..144,700&family=JetBrains+Mono:wght@400;500;700&display=swap');

:root {{
  --bg:{t['bg']}; --card:{t['card']}; --input:{t['input']};
  --border:{t['border']}; --text:{t['text']}; --muted:{t['muted']};
  --green:{t['green']}; --red:{t['red']}; --yellow:{t['yellow']};
  --blue:{t['blue']}; --accent:{t['accent']}; --card2:{t['card2']};
  --gradient:{t['gradient']}; --glow:{glow};
}}

/* ═══ Base canvas with ambient light bloom ═══ */
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stApp"] {{
    background: var(--bg) !important; color: var(--text) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    font-feature-settings: 'cv02','cv03','cv04','cv11','ss01';
    letter-spacing: -0.011em;
    text-rendering: optimizeLegibility;
}}
/* Tabular figures — numbers align in neat columns (premium finance look) */
.sig-meta, .sig-price, .kpi-value, [data-testid="stMetricValue"],
.dataframe, code, .mono, [data-testid="stMetric"] {{
    font-feature-settings: 'tnum' 1, 'cv02','cv03';
    font-variant-numeric: tabular-nums;
}}
[data-testid="stAppViewContainer"]::before {{
    content: ""; position: fixed; inset: 0; pointer-events: none; z-index: 0;
    background: {bg_fx};
}}
[data-testid="stHeader"] {{ background: transparent !important; }}
#MainMenu, footer, header {{ display: none !important; }}
.block-container {{ padding-top: 1.5rem; padding-bottom: 3rem; max-width: 96%; }}
::-webkit-scrollbar {{ width: 6px; height: 6px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 10px; }}
::-webkit-scrollbar-thumb:hover {{ background: var(--accent); }}

/* Numbers always tabular — institutional data discipline */
.card .val, table.t td, .sector-tbl td, .sig-meta, .pick-prices {{
    font-variant-numeric: tabular-nums;
}}

/* ═══ Title with animated accent underline ═══ */
.dash-title {{
    font-size: 2rem; font-weight: 800; padding-bottom: 1rem; margin-bottom: 1.5rem;
    display: flex; align-items: center; justify-content: space-between;
    letter-spacing: -0.03em; border-bottom: 1px solid var(--border);
    position: relative;
}}
.dash-title::after {{
    content: ""; position: absolute; bottom: -1px; left: 0; height: 2px; width: 180px;
    background: linear-gradient(90deg, var(--accent), transparent);
    animation: pulse-line 3s ease-in-out infinite;
}}
@keyframes pulse-line {{ 0%,100% {{ opacity: .5; width: 180px; }} 50% {{ opacity: 1; width: 280px; }} }}
.dash-title-text {{
    background: linear-gradient(to right, var(--text), var(--muted));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}}
.dash-title span.hl {{ color: var(--accent); -webkit-text-fill-color: var(--accent); }}

/* ═══ KPI cards — glass + top shimmer line + lift on hover ═══ */
.cards {{ display: flex; gap: 1.2rem; flex-wrap: wrap; margin-bottom: 2.5rem; }}
.card {{
    background: var(--gradient); border: 1px solid var(--border); border-radius: 16px;
    padding: 1.5rem; flex: 1; min-width: 160px; position: relative; overflow: hidden;
    box-shadow: 0 10px 30px -10px rgba(0,0,0,0.55);
    transition: all .4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px);
}}
.card::before {{
    content: ""; position: absolute; top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, var(--accent), transparent);
    opacity: 0; transition: opacity .4s ease;
}}
.card:hover {{
    transform: translateY(-6px);
    box-shadow: 0 24px 48px -12px rgba(0,0,0,0.7), 0 0 24px var(--glow);
    border-color: var(--accent);
}}
.card:hover::before {{ opacity: 1; }}
.card .lbl {{
    font-size: .72rem; text-transform: uppercase; letter-spacing: .15em;
    color: var(--muted); margin-bottom: .5rem; font-weight: 700;
}}
.card .val {{ font-size: 1.6rem; font-weight: 800; color: var(--text); letter-spacing: -0.03em; }}
.card .sub {{ font-size: .8rem; color: var(--muted); margin-top: .4rem; font-weight: 600; }}

/* ═══ Section headers with gradient rail ═══ */
.green {{ color: var(--green) !important; }} .red {{ color: var(--red) !important; }}
.yellow {{ color: var(--yellow) !important; }} .blue {{ color: var(--blue) !important; }}
.sec {{
    font-size: .95rem; font-weight: 800; text-transform: uppercase;
    letter-spacing: .12em; color: var(--text); margin: 2.5rem 0 1.2rem;
    padding-left: 1rem; position: relative;
}}
.sec::before {{
    content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 4px;
    border-radius: 4px;
    background: linear-gradient(180deg, var(--accent), transparent);
}}

/* ═══ Tables — glass panel + accent header rail + row glow ═══ */
.tbl-wrap {{
    overflow-x: auto; background: var(--card); border: 1px solid var(--border);
    border-radius: 16px; box-shadow: 0 10px 30px -10px rgba(0,0,0,0.55);
    backdrop-filter: blur(16px); margin-bottom: 1.5rem;
}}
table.t, .sector-tbl {{ width: 100%; border-collapse: collapse; font-size: .85rem; }}
table.t th, .sector-tbl th {{
    background: var(--card2); color: var(--muted); text-transform: uppercase;
    letter-spacing: .08em; font-weight: 700; padding: 1.1rem 1rem; text-align: right;
    border-bottom: 1px solid var(--border); position: sticky; top: 0;
}}
table.t th.l, table.t td.l {{ text-align: left; }}
table.t td, .sector-tbl td {{
    padding: .95rem 1rem; border-bottom: 1px solid var(--border);
    text-align: right; color: var(--text); font-weight: 600;
    transition: background .2s ease;
}}
table.t tr:last-child td, .sector-tbl tr:last-child td {{ border-bottom: none; }}
table.t tr:hover td, .sector-tbl tr:hover td {{
    background: linear-gradient(90deg, transparent, var(--glow), transparent);
}}
table.t tr.row-profit td {{ box-shadow: inset 3px 0 0 var(--green); }}
table.t tr.row-loss td   {{ box-shadow: inset 3px 0 0 var(--red); }}

/* ═══ Badges with glow ═══ */
.pos {{ color: var(--green); font-weight: 800; }} .neg {{ color: var(--red); font-weight: 800; }}
.zero-cell {{ color: var(--muted) !important; }}
.badge {{
    display: inline-block; padding: .3rem .8rem; border-radius: 8px;
    font-size: .68rem; font-weight: 800; text-transform: uppercase; letter-spacing: .08em;
}}
.b-open {{ background: color-mix(in srgb, var(--yellow) 10%, transparent); color: var(--yellow);
    border: 1px solid color-mix(in srgb, var(--yellow) 35%, transparent);
    box-shadow: 0 0 12px color-mix(in srgb, var(--yellow) 12%, transparent); }}
.b-cl {{ background: rgba(16,185,129,.1); color: var(--green);
    border: 1px solid rgba(16,185,129,.35); box-shadow: 0 0 12px rgba(16,185,129,.12); }}
.b-cll {{ background: rgba(239,68,68,.1); color: var(--red);
    border: 1px solid rgba(239,68,68,.35); box-shadow: 0 0 12px rgba(239,68,68,.12); }}

/* ═══ Signal / pick / outlook cards — glass + animated entrance ═══ */
.sig-grid, .pick-grid, .outlook-grid {{
    display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
    gap: 1.5rem; margin-top: 1rem;
}}
.sig-card, .pick-card, .outlook-card {{
    background: var(--card); border: 1px solid var(--border); border-radius: 16px;
    padding: 1.5rem; transition: all .3s ease; backdrop-filter: blur(16px);
    box-shadow: 0 10px 20px -5px rgba(0,0,0,0.35);
    animation: card-in .45s ease both;
}}
@keyframes card-in {{ from {{ opacity: 0; transform: translateY(12px); }} to {{ opacity: 1; transform: none; }} }}
.sig-card:hover, .pick-card:hover {{
    transform: translateY(-5px); border-color: var(--accent);
    box-shadow: 0 18px 36px -8px rgba(0,0,0,0.55), 0 0 20px var(--glow);
}}
.sig-card.sell  {{ border-top: 3px solid var(--red); }}
.sig-card.avg   {{ border-top: 3px solid var(--yellow); }}
.sig-card.hold  {{ border-top: 3px solid var(--green); }}
.sig-card.watch {{ border-top: 3px solid var(--muted); }}
.pick-card      {{ border-top: 3px solid var(--accent); }}

.sig-action {{ font-size: .9rem; font-weight: 800; margin-bottom: .8rem;
    text-transform: uppercase; letter-spacing: .1em; }}
.sig-meta, .pick-sector {{ font-size: .8rem; color: var(--muted); font-weight: 600; }}
.sig-reason, .pick-prices {{ font-size: .88rem; margin-top: 1rem; color: var(--text); line-height: 1.65; }}
.sig-price, .pick-reason {{
    font-size: .82rem; margin-top: 1.2rem; padding-top: 1rem;
    border-top: 1px solid var(--border); font-weight: 600; color: var(--muted);
}}
.str-bar {{ height: 4px; border-radius: 2px; margin-top: 1rem; background: var(--input);
    overflow: hidden; }}
.str-fill {{ height: 100%; border-radius: 2px; transition: width .8s cubic-bezier(.22,1,.36,1); }}
.rr-warn {{ font-size: .75rem; color: var(--yellow); font-weight: 700; }}
.news-item {{
    padding: .6rem .9rem; border-left: 3px solid var(--accent); margin-bottom: .5rem;
    font-size: .85rem; background: var(--card2); border-radius: 0 8px 8px 0;
    transition: all .2s ease;
}}
.news-item:hover {{ border-left-width: 6px; background: var(--input); }}

/* ═══ Sidebar, inputs, buttons ═══ */
[data-testid="stSidebar"] {{
    background: var(--card) !important; border-right: 1px solid var(--border);
    padding-top: 1rem;
}}
/* CRITICAL FIX — keep the sidebar expand ("maximize") control ALWAYS visible.
   Streamlit changed this test-id across versions, so we target every known
   variant. The backdrop-filter was removed from the sidebar above because it
   created a stacking context that hid this button after collapsing. */
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapseButton"],
button[kind="headerNoPadding"][data-testid="baseButton-headerNoPadding"] {{
    display: flex !important; visibility: visible !important; opacity: 1 !important;
    z-index: 1000000 !important;
}}
/* The floating expand control when sidebar is collapsed */
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"] {{
    position: fixed !important; top: .55rem !important; left: .55rem !important;
    background: var(--accent) !important; border-radius: 8px !important;
    padding: .3rem !important; box-shadow: 0 2px 12px rgba(0,0,0,.5) !important;
}}
[data-testid="stSidebarCollapsedControl"] svg,
[data-testid="collapsedControl"] svg {{
    color: #000 !important; fill: #000 !important; width: 1.5rem; height: 1.5rem;
}}
[data-testid="stSidebarCollapseButton"] svg {{ color: var(--text) !important; }}
/* The Streamlit top header bar can overlap the control — keep it transparent
   and non-blocking so the expand button is always clickable. */
[data-testid="stHeader"] {{
    background: transparent !important; z-index: 1 !important;
}}
/* Ensure dataframes scroll internally (both axes) and never clip results */
[data-testid="stDataFrame"] {{ overflow: auto !important; }}
[data-testid="stDataFrame"] > div {{ overflow: auto !important; max-width: 100% !important; }}
.stDataFrame [data-testid="stDataFrameResizable"] {{ overflow: auto !important; }}
/* Mobile: bigger tap target, sidebar takes most of the screen when open */
@media (max-width: 768px) {{
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"] {{
        top: .5rem !important; left: .5rem !important;
        padding: .45rem !important; transform: scale(1.2);
    }}
    [data-testid="stSidebar"] {{ min-width: 82vw !important; }}
}}
div[data-baseweb="input"], div[data-baseweb="select"],
[data-testid="stNumberInputContainer"] {{
    background-color: var(--input) !important; border: 1px solid var(--border) !important;
    border-radius: 10px !important; transition: border-color .2s ease !important;
}}
div[data-baseweb="input"]:focus-within, div[data-baseweb="select"]:focus-within,
[data-testid="stNumberInputContainer"]:focus-within {{
    border-color: var(--accent) !important; box-shadow: 0 0 0 2px var(--glow) !important;
}}
div[data-baseweb="input"] input, [data-testid="stNumberInputContainer"] input {{
    color: var(--text) !important; -webkit-text-fill-color: var(--text) !important;
    background-color: transparent !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important; font-weight: 600 !important;
}}
button[data-testid="stNumberInputStepDown"], button[data-testid="stNumberInputStepUp"] {{
    background-color: var(--card2) !important; color: var(--text) !important; border: none !important;
}}
div[role="listbox"] {{ background-color: var(--card2) !important;
    border: 1px solid var(--border) !important; border-radius: 10px !important; }}
ul[role="listbox"] li {{ color: var(--text) !important; font-weight: 500 !important; }}
ul[role="listbox"] li[aria-selected="true"] {{
    background-color: var(--accent) !important; color: #000 !important; font-weight: 800 !important; }}

.stButton>button {{
    background: var(--card2) !important; border: 1px solid var(--border) !important;
    color: var(--text) !important; border-radius: 10px !important; font-weight: 700 !important;
    letter-spacing: .05em !important; padding: .6rem 1.2rem !important;
    transition: all .3s ease !important;
}}
.stButton>button:hover {{
    border-color: var(--accent) !important; background: var(--accent) !important;
    color: #000 !important; box-shadow: 0 0 24px var(--glow) !important;
    transform: translateY(-1px) scale(1.01);
}}

/* ═══ Tabs — underline glide ═══ */
.stTabs [data-baseweb="tab-list"] {{
    background: transparent; gap: 2.2rem; padding: 0 .5rem;
    border-bottom: 1px solid var(--border);
}}
.stTabs [data-baseweb="tab"] {{
    background: transparent; color: var(--muted); font-weight: 700; padding: 1.2rem 0;
    border: none; border-bottom: 3px solid transparent; text-transform: uppercase;
    letter-spacing: .08em; font-size: .82rem; transition: all .3s ease;
}}
.stTabs [data-baseweb="tab"]:hover {{ color: var(--text); }}
.stTabs [aria-selected="true"] {{
    background: transparent !important; color: var(--text) !important;
    border-bottom-color: var(--accent) !important;
    text-shadow: 0 0 18px var(--glow);
}}

/* ═══ Expanders ═══ */
[data-testid="stExpander"] {{
    background-color: var(--card) !important; border: 1px solid var(--border) !important;
    border-radius: 14px !important; margin-bottom: .8rem !important;
    backdrop-filter: blur(12px);
}}
[data-testid="stExpander"] summary p {{ font-weight: 700 !important; color: var(--text) !important; }}

/* ═══ Regime banner — live pulse dot + glass ═══ */
.refresh-badge {{
    display: inline-flex; align-items: center; gap: .5rem;
    background: rgba(16, 185, 129, 0.1); color: var(--green);
    padding: .4rem 1rem; border-radius: 30px; font-size: .72rem; font-weight: 800;
    border: 1px solid rgba(16, 185, 129, 0.4); letter-spacing: .1em;
    text-transform: uppercase; box-shadow: 0 0 15px rgba(16, 185, 129, 0.2);
}}
.refresh-badge::before {{
    content: ""; width: 7px; height: 7px; border-radius: 50%;
    background: var(--green); animation: live-pulse 1.8s ease-in-out infinite;
}}
@keyframes live-pulse {{
    0%, 100% {{ opacity: 1; box-shadow: 0 0 0 0 rgba(16,185,129,.5); }}
    50% {{ opacity: .6; box-shadow: 0 0 0 5px rgba(16,185,129,0); }}
}}
.regime-banner {{
    border-radius: 16px; padding: 1.2rem 1.8rem; display: flex; align-items: center;
    gap: 1.2rem; margin-bottom: 2.5rem; flex-wrap: wrap;
    box-shadow: 0 15px 35px -10px rgba(0,0,0,0.6); border: 1px solid var(--border);
    backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px);
}}

/* ════════════════════════════════════════════════════════════════════
   ✦ PREMIUM POLISH LAYER — refined typography, buttons, inputs, sidebar
   ════════════════════════════════════════════════════════════════════ */

/* Display serif for major headings — adds an editorial, private-bank feel.
   (Fraunces is already loaded in the main @import at the top of this stylesheet.) */
.dash-title, .sec {{
    font-family: 'Fraunces', Georgia, serif !important;
    font-weight: 600 !important; letter-spacing: -0.02em !important;
}}

/* Section headers get a refined gold tick + tighter rhythm */
.sec {{
    font-size: 1.35rem !important; font-weight: 600 !important;
    margin: 1.8rem 0 1.1rem !important; padding-left: .9rem !important;
    position: relative; line-height: 1.2;
}}
.sec::before {{
    content: ""; position: absolute; left: 0; top: 50%; transform: translateY(-50%);
    width: 4px; height: 70%; border-radius: 3px;
    background: linear-gradient(180deg, var(--accent), transparent);
}}

/* ✦ Buttons — premium gradient, lift, gold glow on hover */
.stButton > button, .stDownloadButton > button {{
    background: var(--card2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    font-weight: 600 !important; font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    letter-spacing: -0.01em !important;
    transition: all .25s cubic-bezier(0.175,0.885,0.32,1.275) !important;
}}
.stButton > button:hover, .stDownloadButton > button:hover {{
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px -8px var(--glow) !important;
}}
.stButton > button:active, .stDownloadButton > button:active {{
    transform: translateY(0) !important;
}}
/* Primary buttons (form submit) get the gold fill */
.stButton > button[kind="primary"], .stForm button[kind="primaryFormSubmit"] {{
    background: linear-gradient(145deg, var(--accent), var(--accent)) !important;
    color: var(--bg) !important; border: none !important;
    box-shadow: 0 4px 20px -6px var(--glow) !important;
}}
.stButton > button[kind="primary"]:hover {{
    color: var(--bg) !important; filter: brightness(1.08);
}}

/* ✦ Inputs / selects — frosted with gold focus ring */
.stTextInput input, .stNumberInput input, .stDateInput input,
[data-baseweb="select"] > div, [data-baseweb="input"] > div {{
    background: var(--card2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    transition: border-color .2s, box-shadow .2s !important;
}}
.stTextInput input:focus, .stNumberInput input:focus, .stDateInput input:focus {{
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--glow) !important;
}}

/* ✦ Tabs — underline slides, gold active */
.stTabs [data-baseweb="tab-list"] {{ gap: .3rem; border-bottom: 1px solid var(--border); }}
.stTabs [data-baseweb="tab"] {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    font-weight: 600 !important; color: var(--muted) !important;
    border-radius: 8px 8px 0 0 !important; transition: color .2s !important;
}}
.stTabs [aria-selected="true"] {{ color: var(--accent) !important; }}
.stTabs [data-baseweb="tab-highlight"] {{ background: var(--accent) !important; }}

/* ✦ Sidebar — deeper glass, refined nav radio */
[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, rgba(0,0,0,.25), transparent), var(--card) !important;
    border-right: 1px solid var(--border) !important;
    backdrop-filter: blur(24px) !important; -webkit-backdrop-filter: blur(24px) !important;
}}
[data-testid="stSidebar"] .stRadio label {{
    transition: all .2s !important; border-radius: 8px !important;
    padding: .15rem .4rem !important;
}}
[data-testid="stSidebar"] .stRadio label:hover {{
    background: var(--glow) !important;
}}

/* ✦ Metric cards (st.metric) — give them the glass treatment */
[data-testid="stMetric"] {{
    background: var(--gradient) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important; padding: 1rem 1.2rem !important;
    box-shadow: 0 8px 24px -12px rgba(0,0,0,.5) !important;
    transition: transform .3s, border-color .3s !important;
}}
[data-testid="stMetric"]:hover {{
    transform: translateY(-3px) !important; border-color: var(--accent) !important;
}}
[data-testid="stMetricValue"] {{
    font-variant-numeric: tabular-nums !important; letter-spacing: -.02em !important;
}}

/* ✦ Dataframes — softer, rounded, bordered */
[data-testid="stDataFrame"] {{
    border: 1px solid var(--border) !important; border-radius: 12px !important;
    overflow: hidden !important;
}}

/* ✦ Expanders — refined */
[data-testid="stExpander"] {{
    border: 1px solid var(--border) !important; border-radius: 12px !important;
    background: var(--card) !important; overflow: hidden;
}}

/* ✦ Subtle slide-in on load — NO opacity fade, so a page switch never shows the
   previous page through the new one (the old "foggy ghost" effect). */
.block-container > div {{ animation: viewslide .22s ease both; }}
@keyframes viewslide {{ from {{ transform: translateY(4px); }} to {{ transform: none; }} }}

/* Respect reduced motion */
@media (prefers-reduced-motion: reduce) {{
    *, ::before, ::after {{ animation: none !important; transition: none !important; }}
}}
</style>
"""

# ── Price Fetcher & Logic ───────────────────────────────────────────────────────
_CACHE = {}
_TTL = 300


def fetch_price(symbol):
    clean = str(symbol).upper().strip()
    for sfx in [".NS", ".BO", ".NSE", ".BSE"]:
        if clean.endswith(sfx):
            clean = clean[:-len(sfx)]
    if clean in _CACHE and time.time() - _CACHE[clean][1] < _TTL:
        return _CACHE[clean][0]

    def _fast(t):
        try:
            fi = t.fast_info
        except Exception:
            return None
        for key in ("last_price", "lastPrice", "regularMarketPrice"):
            for getter in (lambda: fi.get(key) if hasattr(fi, "get") else None,
                           lambda: getattr(fi, key, None),
                           lambda: fi[key]):
                try:
                    v = getter()
                    if v is not None and not pd.isna(v) and float(v) > 0:
                        return float(v)
                except Exception:
                    pass
        return None

    for sfx in [""]:
        try:
            t = yf.Ticker(clean + sfx)
            v = _fast(t)
            if v is not None:
                p = round(v, 2)
                _CACHE[clean] = (p, time.time())
                return p
            # auto_adjust=False → ACTUAL close, not dividend/split back-adjusted
            h = t.history(period="5d", interval="1d", auto_adjust=False)
            if h is not None and not h.empty and "Close" in h.columns:
                lv = h["Close"].dropna()
                if not lv.empty:
                    p = round(float(lv.iloc[-1]), 2)
                    _CACHE[clean] = (p, time.time())
                    return p
        except Exception:
            continue
    return None


@st.cache_data(ttl=120, show_spinner=False)
def fetch_chart_data(symbol, period, interval):
    """Fetch OHLCV for the chart at any interval (intraday or daily).
    Yahoo intraday limits: 1m→7d max, 5m/15m→60d max, 1h→730d.
    auto_adjust=False for accurate actual prices. Returns DataFrame or None."""
    import yfinance as _yf
    clean = str(symbol).upper().strip()
    for sfx in [".NS", ".BO", ".NSE", ".BSE"]:
        if clean.endswith(sfx):
            clean = clean[:-len(sfx)]
    for sfx in [""]:
        for _attempt in range(2):
            try:
                t = _yf.Ticker(clean + sfx)
                df = t.history(period=period, interval=interval, auto_adjust=False)
                if df is not None and not df.empty and "Close" in df.columns:
                    return df.dropna(subset=["Close"])
            except Exception:
                pass
            try:
                df = _yf.download(clean + sfx, period=period, interval=interval,
                                  auto_adjust=False, progress=False, threads=False)
                if df is not None and not df.empty:
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    if "Close" in df.columns:
                        return df.dropna(subset=["Close"])
            except Exception:
                pass
            time.sleep(0.3)
    return None


# Valid (period, interval) combinations for Yahoo intraday data
_CHART_TIMEFRAMES = {
    "5 min":  ("5d",  "5m"),
    "15 min": ("1mo", "15m"),
    "1 hour": ("3mo", "1h"),
    "Daily":  ("6mo", "1d"),
    "Weekly": ("2y",  "1wk"),
}


def enrich(df):
    if df.empty:
        return df
    # Use cached price fetcher — avoids re-fetching on every Streamlit rerun
    symbols = tuple(sorted(df["stock"].unique().tolist()))
    prices  = _cached_prices(symbols)
    df = df.copy()
    df["cmp"] = df["stock"].map(prices)
    df["nse_label"] = df["stock"]
    df["invested"] = df["quantity"] * df["buy_at"]
    df["current_amt"] = __import__("numpy").where(
        df["status"] == "Open",
        df["quantity"] * df["cmp"].fillna(df["buy_at"]),
        df["quantity"] * df["sell_at"].fillna(df["buy_at"])
    )
    df["total_amt"] = __import__("numpy").where(
        df["sell_at"].notna(),
        df["quantity"] * df["sell_at"],
        df["current_amt"]
    )
    df["profit"] = df["total_amt"] - df["invested"]
    df["profit_pct"] = (df["profit"] / df["invested"] * 100).round(2)
    return df


def calc_analytics(df):
    if df.empty:
        return {}
    closed = df[df["status"] == "Closed"]
    open_t = df[df["status"] == "Open"]
    total_closed = len(closed)
    wins   = len(closed[closed["profit"] > 0]) if not closed.empty else 0
    losses = len(closed[closed["profit"] < 0]) if not closed.empty else 0
    win_rate  = (wins / total_closed * 100) if total_closed > 0 else 0
    avg_win   = closed[closed["profit"] > 0]["profit"].mean() if wins > 0 else 0
    avg_loss  = abs(closed[closed["profit"] < 0]["profit"].mean()) if losses > 0 else 0
    exp = ((wins / total_closed if total_closed > 0 else 0) * avg_win) - \
          ((losses / total_closed if total_closed > 0 else 0) * avg_loss)
    gp = closed[closed["profit"] > 0]["profit"].sum() if wins > 0 else 0
    gl = abs(closed[closed["profit"] < 0]["profit"].sum()) if losses > 0 else 1
    max_dd = abs(
        (closed["profit"].cumsum() - closed["profit"].cumsum().expanding().max()).min()
    ) if not closed.empty else 0
    avg_hold = round(
        (pd.to_datetime(closed["closed_date"]) -
         pd.to_datetime(closed["added_date"])).dt.days.mean(), 1
    ) if not closed.empty and "closed_date" in closed.columns else 0
    sharpe = (closed["profit_pct"].mean() / closed["profit_pct"].std()
              if not closed.empty and closed["profit_pct"].std() > 0 else 0)
    return {
        "total_trades": len(df), "closed_trades": total_closed,
        "open_trades": len(open_t), "wins": wins, "losses": losses,
        "win_rate": round(win_rate, 1), "avg_win": round(avg_win, 0),
        "avg_loss": round(avg_loss, 0), "expectancy": round(exp, 0),
        "profit_factor": round(gp / gl, 2), "max_drawdown": round(max_dd, 0),
        "avg_hold_days": avg_hold, "sharpe": round(sharpe, 2)
    }


# ── Formatting helpers ─────────────────────────────────────────────────────────
def fi(v):   return f"${v:,.0f}"    if not pd.isna(v) else "—"
def fi2(v):  return f"${v:,.2f}"   if not pd.isna(v) else "—"
def fp(v):   return f"{'+' if v >= 0 else ''}{v:.2f}%" if not pd.isna(v) else "—"

def cv_cell(v, fn):
    if pd.isna(v):
        return f"<td>{fn(v)}</td>"
    if v > 0:
        return f'<td class="profit-cell pos">{fn(v)}</td>'
    if v < 0:
        return f'<td class="profit-cell neg">{fn(v)}</td>'
    return f'<td class="zero-cell">{fn(v)}</td>'

def badge(status, profit=None):
    if status == "Open":
        return '<span class="badge b-open">Open</span>'
    if profit is not None and profit < 0:
        return '<span class="badge b-cll">Closed ✗</span>'
    return '<span class="badge b-cl">Closed ✓</span>'

def card(lbl, val, sub="", cls=""):
    sub_html = f'<div class="sub">{sub}</div>' if sub else ""
    return f'<div class="card"><div class="lbl">{lbl}</div><div class="val {cls}">{val}</div>{sub_html}</div>'


# ── Chart helpers ──────────────────────────────────────────────────────────────
def base_layout(fig, title):
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#f8fafc", weight="bold"), x=.01),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cbd5e1", size=11), margin=dict(l=8, r=8, t=45, b=8))
    fig.update_xaxes(gridcolor="#334155", zerolinecolor="#334155", tickfont=dict(color="#cbd5e1"))
    fig.update_yaxes(gridcolor="#334155", zerolinecolor="#334155", tickfont=dict(color="#cbd5e1"))
    return fig


def chart_alloc(df):
    g = df.groupby("stock")["invested"].sum().reset_index()
    return base_layout(go.Figure(go.Pie(
        labels=g["stock"], values=g["invested"], hole=0.4,
        marker=dict(colors=px.colors.qualitative.Dark24,
                    line=dict(color="rgba(0,0,0,0)", width=0)),
        textinfo="percent+label", textfont=dict(size=11, color="#ffffff")
    )), "Portfolio Allocation")


def chart_pnl(df):
    d = df.sort_values("profit")
    fig = base_layout(go.Figure(go.Bar(
        x=d["profit"], y=d["stock"], orientation="h",
        marker=dict(color=["#ef4444" if v < 0 else "#10b981" for v in d["profit"]],
                    line=dict(width=0)),
        text=[fp(p) for p in d["profit_pct"]],
        textposition="outside", textfont=dict(color="#f8fafc", size=10)
    )), "P&L by Stock")
    fig.update_layout(showlegend=False, margin=dict(l=8, r=55, t=45, b=8))
    fig.update_xaxes(tickprefix="$")
    return fig


def chart_donut(df):
    c = df["status"].value_counts().reset_index()
    c.columns = ["Status", "Count"]
    fig = base_layout(go.Figure(go.Pie(
        labels=c["Status"], values=c["Count"], hole=.6,
        marker=dict(
            colors=[{"Open": "#f59e0b", "Closed": "#10b981"}.get(s, "#94a3b8")
                    for s in c["Status"]],
            line=dict(color="rgba(0,0,0,0)", width=0)),
        textinfo="percent+value", textfont=dict(size=12, color="#ffffff")
    )), "Open vs Closed")
    fig.add_annotation(
        text=f"<b>{len(df)}</b><br><span style='font-size:10px'>TRADES</span>",
        font=dict(size=18, color="#f8fafc"), showarrow=False, x=.5, y=.5)
    return fig


def chart_growth(hist, cur_val, cur_inv):
    today = datetime.now().strftime("%Y-%m-%d")
    rows = hist[["snapshot_date", "total_invested", "current_value"]].to_dict("records")
    if not hist.empty and hist.iloc[-1]["snapshot_date"] != today:
        rows.append({"snapshot_date": today,
                     "total_invested": cur_inv, "current_value": cur_val})
    elif hist.empty:
        rows = [{"snapshot_date": today,
                 "total_invested": cur_inv, "current_value": cur_val}]
    d = pd.DataFrame(rows)
    fig = base_layout(go.Figure([
        go.Scatter(x=pd.to_datetime(d["snapshot_date"]), y=d["current_value"],
                   name="Value", line=dict(color="#10b981", width=3),
                   fill="tozeroy", fillcolor="rgba(16,185,129,0.1)"),
        go.Scatter(x=pd.to_datetime(d["snapshot_date"]), y=d["total_invested"],
                   name="Invested", line=dict(color="#3b82f6", width=2, dash="dash"))
    ]), "Portfolio Growth")
    fig.update_layout(hovermode="x unified")
    fig.update_yaxes(tickprefix="$")
    return fig


# ── Signal card renderer ──────────────────────────────────────────────────────
def _fmt_rr(rr):
    if rr is None:
        return "—"
    if rr > 10:
        return f'<span class="rr-warn">⚠️ {rr} (verify ATR)</span>'
    if rr > 5:
        return f'<span style="color:#f59e0b;font-weight:700">{rr}</span>'
    return str(rr)


def render_signals(signals, theme_t):
    if not signals:
        st.info("No signals available.")
        return

    html = '<div class="sig-grid">'
    for s in signals:
        action = s.get("action", "")
        c = ("sell"  if "SELL"    in action else
             "avg"   if "AVERAGE" in action else
             "hold"  if "HOLD"    in action else "watch")
        clr = (theme_t["red"]    if c == "sell"  else
               theme_t["yellow"] if c == "avg"   else
               theme_t["green"]  if c == "hold"  else theme_t["muted"])

        cmp_v   = s.get("cmp")
        rsi_v   = s.get("rsi")
        pct     = s.get("pct_from_buy")
        target  = s.get("target")
        sl      = s.get("stop_loss")
        rr      = s.get("risk_reward")
        trend_v = s.get("trend", "—")
        macd_v  = s.get("macd_signal", "—")
        reason  = s.get("reason", "")
        strength = s.get("strength", 30)

        cmp_str = f"${cmp_v}" if cmp_v is not None else "—"
        rsi_str = str(rsi_v)  if rsi_v is not None else "—"
        pct_str = f"{pct:+.1f}%" if pct is not None else "—%"
        tgt_str = f"${target}" if target is not None else "—"
        sl_str  = f"${sl}"    if sl  is not None else "—"
        rr_html = _fmt_rr(rr)

        if c == "sell":
            ph = (f"🎯 Exit: {tgt_str} | 🛑 Re-entry: {sl_str}<br>"
                  f"📉 {trend_v} | MACD: {macd_v}")
        elif c == "avg":
            avg_p   = s.get("avg_price")
            new_avg = s.get("new_avg")
            new_sl  = s.get("new_sl")
            ph = (f"💰 Avg: {'$'+str(avg_p) if avg_p else '—'} | "
                  f"New Avg: {'$'+str(new_avg) if new_avg else '—'}<br>"
                  f"🛑 SL: {'$'+str(new_sl) if new_sl else '—'} | 🎯 Target: {tgt_str}")
        else:
            ph = (f"🎯 Target: {tgt_str} | 🛑 SL: {sl_str}<br>"
                  f"📊 R:R {rr_html} | {trend_v}")

        # ── Build badges as clean single-line strings (no multi-line f-string
        #    expressions, which can break Streamlit's HTML rendering) ──────────
        badge_new = ""
        if s.get("limited_history"):
            badge_new = (f'<span style="font-size:.62rem;background:rgba(245,158,11,.15);'
                         f'color:#f59e0b;padding:.1rem .4rem;border-radius:4px;'
                         f'font-weight:700;margin-left:.3rem">🆕 {s.get("bars","")}d history</span>')

        badge_vcp = ""
        if s.get("vcp"):
            _vq = s.get("vcp_quality") or ""
            _vr = "▸READY" if s.get("vcp_ready") else ""
            badge_vcp = (f'<span style="font-size:.62rem;background:rgba(16,185,129,.18);'
                         f'color:#10b981;padding:.1rem .4rem;border-radius:4px;'
                         f'font-weight:700;margin-left:.3rem">🎯 VCP {_vq}{_vr}</span>')

        badge_rs = ""
        _rsr = s.get("rs_ratio")
        if s.get("rs_outperforming") and isinstance(_rsr, (int, float)):
            badge_rs = (f'<span style="font-size:.62rem;background:rgba(59,130,246,.18);'
                        f'color:#3b82f6;padding:.1rem .4rem;border-radius:4px;'
                        f'font-weight:700;margin-left:.3rem">💪 RS {_rsr:.2f}</span>')

        badges = badge_new + badge_vcp + badge_rs

        html += f"""
<div class="sig-card {c}">
  <div class="sig-action" style="color:{clr}">{action}</div>
  <div style="font-size:.9rem;font-weight:800;margin-bottom:.3rem">
    {s.get('stock','')}
    <span style="font-size:.7rem;color:var(--muted);font-weight:400">{s.get('sector','')}</span>
    {badges}
  </div>
  <div class="sig-meta">CMP {cmp_str} · RSI {rsi_str} · {pct_str}</div>
  <div class="sig-reason">{reason}</div>
  <div class="sig-price">{ph}</div>
  <div class="str-bar">
    <div class="str-fill" style="width:{strength}%;background:{clr}"></div>
  </div>
</div>"""

    # Collapse leading whitespace on each line — prevents Streamlit's markdown
    # parser from ever treating indented HTML as a code block (which would show
    # the raw <span> tags as literal text).
    html_clean = "\n".join(line.lstrip() for line in (html + "</div>").split("\n"))
    st.markdown(html_clean, unsafe_allow_html=True)


# ── Sector table renderer ─────────────────────────────────────────────────────
def render_sector(sdf, t):
    if sdf is None or sdf.empty:
        return

    def medal(rank):
        return "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "📊"

    rows = ""
    for _, r in sdf.iterrows():
        rs = r.get("rs_vs_nifty_1m", 0) or 0
        rs_clr = "#10b981" if rs > 0 else "#ef4444"
        avg_rsi = r["avg_rsi"]
        rsi_str = f"{avg_rsi:.0f}" if avg_rsi and not pd.isna(avg_rsi) else "—"
        rows += (
            f"<tr>"
            f"<td style='font-weight:700'>{medal(r['rank'])} #{int(r['rank'])}</td>"
            f"<td><b style='font-size:.9rem'>{r['sector']}</b></td>"
            f"<td style='color:var(--muted);font-size:.75rem'>{r['stocks']}</td>"
            f"<td style='text-align:center;font-weight:600'>{rsi_str}</td>"
            f"<td style='text-align:right'>"
            f"  <span class='{'pos' if r['avg_pct'] > 0 else 'neg'}'>"
            f"  {r['avg_pct']:+.1f}%</span></td>"
            f"<td style='text-align:right;font-size:.8rem'>"
            f"  <span style='color:{rs_clr};font-weight:700'>{rs:+.1f}%</span></td>"
            f"<td style='font-size:.8rem;font-weight:600'>"
            f"  {r.get('rrg_quadrant', '—')}</td>"
            f"<td>"
            f"  <div style='background:{t['input']};height:6px;width:100%;border-radius:4px'>"
            f"    <div style='background:{t['accent']};"
            f"         width:{min(r['momentum_score'] * 100, 100):.0f}%;"
            f"         height:6px;border-radius:4px'></div></div>"
            f"  <span style='font-size:.75rem;font-weight:600'>"
            f"  {r['momentum_score']:.2f}</span></td>"
            f"</tr>"
        )

    st.markdown(
        f'<div class="tbl-wrap"><table class="sector-tbl">'
        f'<thead><tr>'
        f'<th>Rank</th><th>Sector</th><th>Top Movers</th>'
        f'<th style="text-align:center">RSI</th>'
        f'<th style="text-align:right">1M Chg</th>'
        f'<th style="text-align:right">vs S&P</th>'
        f'<th>RRG</th>'
        f'<th>Momentum</th>'
        f'</tr></thead><tbody>{rows}</tbody></table></div>',
        unsafe_allow_html=True
    )


def render_outlook(odf, t):
    if odf is None or odf.empty:
        return
    cards_html = ""
    for _, r in odf.iterrows():
        outlook = r["outlook"]
        clr = t["green"] if any(x in outlook for x in ["Bullish", "Power", "Strong"]) \
              else t["red"]
        avg_rsi = r.get("avg_rsi")
        rsi_str = f"{avg_rsi:.0f}" if avg_rsi and not pd.isna(avg_rsi) else "—"
        cards_html += (
            f"<div class='outlook-card'>"
            f"<div style='font-weight:700;margin-bottom:.3rem'>{r['sector']}</div>"
            f"<div style='color:{clr};font-weight:800;font-size:.9rem'>{outlook}</div>"
            f"<div class='sig-meta' style='margin-top:.4rem'>"
            f"Conf: {r['confidence']}% · Mom: {r['momentum']:.2f}<br>"
            f"RSI: {rsi_str} · Chg: {r['avg_pct']:+.1f}%</div>"
            f"</div>"
        )
    st.markdown(f'<div class="outlook-grid">{cards_html}</div>', unsafe_allow_html=True)


def render_picks(picks, t):
    if not picks:
        return
    cards_html = ""
    for p in picks:
        brd = t["green"] if p["score"] >= 70 else t["yellow"] if p["score"] >= 55 else t["muted"]
        cards_html += (
            f"<div class='pick-card' style='border-top-color:{brd}'>"
            f"<div style='font-weight:800'>{p['stock']} "
            f"<span class='pick-sector'>{p['sector']}</span></div>"
            f"<div style='font-size:.8rem;color:var(--muted);font-weight:600;margin-top:3px'>"
            f"CMP ${p['cmp']} · RSI {p['rsi']} · {p['trend']}</div>"
            f"<div class='pick-prices'>"
            f"🎯 Entry: ${p['entry']}<br>"
            f"🚀 Target: ${p['target']}<br>"
            f"🛑 SL: ${p['stop_loss']}<br>"
            f"📊 R:R: {_fmt_rr(p['risk_reward'])} · Score: {p['score']}</div>"
            f"<div class='pick-reason'>{p['reason']}</div>"
            f"</div>"
        )
    st.markdown(f'<div class="pick-grid">{cards_html}</div>', unsafe_allow_html=True)


# ── News renderer ─────────────────────────────────────────────────────────────
def render_news(news_list):
    if not news_list:
        st.info("No recent news found for your current holdings.")
        return
    for item in news_list:
        st.markdown(
            f'<div class="news-item">{item}</div>',
            unsafe_allow_html=True
        )


# ── Score dashboard ────────────────────────────────────────────────────────────
def render_score_dashboard():
    scores = [
        ("RSI (Wilder's)",          9, "adjust=False + explicit 100/0 edge case — matches TradingView"),
        ("MACD",                    9, "Single-pass crossover, adjust=False, histogram + momentum flags"),
        ("Bollinger Bands",         8, "bb_pos clamped [0,1], bandwidth + squeeze + breakout flags"),
        ("ATR",                     9, "Wilder's EWM smoothing — stops now match Zerodha/TV"),
        ("Supertrend",              9, "Numpy array loop, Wilder ATR(10), mult 2.5 for NSE swing"),
        ("VWAP",                    8, "20-day rolling VWAP + price_vs_vwap % deviation"),
        ("EMA / Trend",             8, "Slope flags (rising/flattening), momentum-fading label, EMA200 back"),
        ("Fibonacci",               8, "Swing-peak based via scipy with degenerate-swing fallback"),
        ("Chart Patterns",          8, "Neckline + Cup&Handle + vol gates"),
        ("Candlesticks",            8, "3-candle patterns, range normalization"),
        ("Signal RR Engine",        9, "Unified _calc_risk_params with PICK mode — zero phantom RR"),
        ("Sector Rotation",         8, "RRG quadrant + RS vs S&P"),
        ("News Engine",             8, "yfinance v1.4 + RSS fallback"),
        ("Liquidity Gate",          8, "Soft gate: liquidity_ok flag, ⚠️ shown on signal, gated for new picks"),
        ("Unified Risk Engine",     9, "Scanner, picks, and portfolio signals all use one engine"),
        ("Bull/Bear Trap Scanner",  9, "5-factor confluence: geometry · volume quality · RSI extreme · Supertrend · reversal candle. Proactive sweep of full US universe."),
        ("Smart Money Concepts",    8, "FVG · Order Blocks · Liquidity Pools · Premium/Discount · Displacement. NSE circuit-filter aware, ATR-normalised thresholds."),
        ("VCP (Volatility Contraction)", 8, "Minervini base detection: 2-4 tightening contractions, volume dry-up, pivot proximity, A+/A/B/C quality grading. Pivot-ready flag + dedicated scanner."),
        ("Relative Strength vs S&P",   8, "IBD-style RS ratio (multi-period weighted) + 1-99 percentile rating across universe. Leaders boost conviction; laggards penalised. Dedicated RS Leaders ranking."),
    ]
    avg = sum(s[1] for s in scores) / len(scores)

    st.markdown(f"""
    <div style="background:var(--card);border:1px solid var(--border);border-radius:12px;
         padding:1.2rem 1.5rem;margin-bottom:1.5rem">
      <div style="font-size:.75rem;color:var(--muted);text-transform:uppercase;
           letter-spacing:.08em;margin-bottom:.5rem">signals.py — Overall Score</div>
      <div style="font-size:2.5rem;font-weight:800;color:var(--accent)">{avg:.1f}<span
           style="font-size:1rem;color:var(--muted);font-weight:400"> / 10</span></div>
      <div style="font-size:.8rem;color:var(--muted);margin-top:.3rem">
        Core engine v12 (Wilder ATR/RSI, numpy Supertrend, 20-day VWAP, swing-peak
        Fibonacci, unified risk engine) plus momentum stack: Trap detection, Smart
        Money Concepts, VCP base detection, and Relative Strength leadership ranking.
      </div>
    </div>
    """, unsafe_allow_html=True)

    rows_html = ""
    for name, score, note in scores:
        fill = min(score * 10, 100)
        clr = ("#10b981" if score >= 8 else "#f59e0b" if score >= 6 else "#ef4444")
        rows_html += f"""
<div style="margin-bottom:.8rem">
  <div style="display:flex;justify-content:space-between;align-items:center;
       margin-bottom:.3rem">
    <span style="font-size:.85rem;font-weight:600;color:var(--text)">{name}</span>
    <span style="font-size:.85rem;font-weight:800;color:{clr}">{score}/10</span>
  </div>
  <div style="background:var(--input);height:5px;border-radius:3px;margin-bottom:.3rem">
    <div style="background:{clr};width:{fill}%;height:5px;border-radius:3px"></div>
  </div>
  <div style="font-size:.75rem;color:var(--muted)">{note}</div>
</div>"""

    st.markdown(
        f'<div style="background:var(--card);border:1px solid var(--border);'
        f'border-radius:12px;padding:1.2rem 1.5rem">{rows_html}</div>',
        unsafe_allow_html=True
    )


# ── Load & Enrich Data ─────────────────────────────────────────────────────────
raw = get_trades(UID)
df  = enrich(raw) if not raw.empty else raw.copy()

if (st.session_state.last_refresh is None or
        (datetime.now() - st.session_state.last_refresh).seconds >= _TTL):
    st.session_state.last_refresh = datetime.now()

# ── Tiered background scan ─────────────────────────────────────────────────────
# FAST TIER (every 5 min):  portfolio signals + news (the core dashboard)
# DEEP TIER (every 15 min): sector rotation + picks + universe scanner + SMC scan
#
# CRITICAL LOAD-ORDER FIX:
# On first login the dashboard must RENDER FIRST, then scan. Otherwise the page
# blocks on the full deep scan (can be minutes) before login even completes.
# We use a `first_render_done` flag: the very first run after login skips all
# scanning, renders the dashboard immediately, and schedules a rerun. From the
# 2nd run onward the scans fire normally in the background.
_now = time.time()

open_raw = raw[raw["status"] == "Open"] if not raw.empty else pd.DataFrame()
_trade_hash = (hash(tuple(sorted(open_raw["id"].tolist())))
               if not open_raw.empty else 0)
_trades_changed = (_trade_hash != st.session_state._trade_hash)

# User-configurable intervals (seconds) and auto-scan toggles
_fast_interval = st.session_state.get("fast_interval_sec", 300)   # default 5 min
_deep_interval = st.session_state.get("deep_interval_sec", 900)   # default 15 min
_auto_fast = st.session_state.get("auto_fast", True)
_auto_deep = st.session_state.get("auto_deep", True)

if not st.session_state.get("first_render_done", False):
    # PASS 1 — first paint after login: render immediately, defer ALL scanning.
    st.session_state.first_render_done = True
    _fast_due = False
    _deep_due = False
    st.session_state._kickoff_scan = True
    st.session_state._scan_stage = "fast"
elif st.session_state.get("_scan_stage") == "fast":
    # PASS 2 — fast scan only (signals + news). Then kick off the deep sequence.
    _fast_due = True
    _deep_due = True
    st.session_state._deep_running = True
    st.session_state._deep_stage = "sector"
    st.session_state._scan_stage = "done"
elif st.session_state.get("_deep_running", False):
    # Deep scan mid-sequence — keep advancing (handled in post-render block).
    _fast_due = False
    _deep_due = True
else:
    # Steady state — scans fire only on their configured schedules (if auto on).
    st.session_state._scan_stage = "done"
    _fast_due = (_auto_fast and
                 (st.session_state.last_auto_scan == 0.0 or
                  (_now - st.session_state.last_auto_scan) >= _fast_interval))
    # Deep auto-trigger: start a fresh sequence when interval elapses
    _deep_elapsed = (st.session_state.last_slow_scan == 0.0 or
                     (_now - st.session_state.last_slow_scan) >= _deep_interval)
    _deep_due = False
    if _auto_deep and _deep_elapsed:
        st.session_state._deep_running = True
        st.session_state._deep_stage = "sector"
        _deep_due = True
    # Manual deep-scan request starts the sequence too
    if st.session_state.get("_manual_deep_request", False):
        st.session_state._manual_deep_request = False
        st.session_state._deep_running = True
        st.session_state._deep_stage = "sector"
        _deep_due = True
    # Manual fast-scan request
    if st.session_state.get("_manual_fast_request", False):
        st.session_state._manual_fast_request = False
        _fast_due = True

if _fast_due:
    n_open = len(open_raw)
    _spinner_msg = (f"🔔 Refreshing {n_open} signal{'s' if n_open!=1 else ''}…"
                    if n_open > 0 else "🔔 Refreshing market data…")
    with st.spinner(_spinner_msg):
        try:
            if _trades_changed or st.session_state.signals_cache is None:
                st.session_state.signals_cache = (
                    generate_signals(open_raw) if not open_raw.empty else [])
                st.session_state.news_cache = (
                    fetch_portfolio_news(open_raw) if not open_raw.empty else [])
                st.session_state._trade_hash = _trade_hash
            st.session_state.last_auto_scan = _now
        except Exception as _e:
            st.session_state.last_auto_scan = _now
            st.toast(f"⚠️ Signal refresh error: {_e}", icon="⚠️")

if _deep_due:
    # Deep scan is due — but we DEFER its execution to the very END of the script
    # (after the whole page renders) so it never blocks or interrupts your view.
    # The actual stage execution happens in the post-render block at the bottom.
    st.session_state._run_deep_now = True
else:
    st.session_state._run_deep_now = False


# ── Portfolio metrics ──────────────────────────────────────────────────────────
if not df.empty:
    odf = df[df["status"] == "Open"]
    cdf = df[df["status"] == "Closed"]
    # Invested & current (portfolio) value reflect ONLY open positions — money
    # tied up in stocks you still hold. Sold/closed positions are excluded
    # because that capital has been freed up (their result lives in realized P&L).
    t_inv    = odf["invested"].sum()    if not odf.empty else 0
    t_cur    = odf["current_amt"].sum() if not odf.empty else 0
    t_real   = cdf["profit"].sum()   if not cdf.empty else 0   # realized (closed)
    t_unreal = odf["profit"].sum()   if not odf.empty else 0   # unrealized (open)
    t_pnl    = t_real + t_unreal       # total P&L across realized + unrealized
    # Return % is measured against invested capital. Use total cost basis
    # (open invested + the original cost of closed trades) so realized gains
    # aren't divided by zero when everything is sold.
    _closed_cost = cdf["invested"].sum() if not cdf.empty else 0
    _pnl_base = t_inv + _closed_cost
    t_pnl_pct = t_pnl / _pnl_base * 100 if _pnl_base > 0 else 0
    best  = df.loc[df["profit_pct"].idxmax(), "stock"]
    worst = df.loc[df["profit_pct"].idxmin(), "stock"]
    save_snapshot(UID, t_inv, t_cur)
else:
    odf = cdf = pd.DataFrame()
    t_inv = t_cur = t_real = t_unreal = t_pnl = t_pnl_pct = 0
    best = worst = "—"

theme_t = THEMES[st.session_state.theme]
st.markdown(theme_css(theme_t), unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        f'<div style="font-size:.85rem;font-weight:800;color:var(--accent);'
        f'margin-bottom:1rem">👤 {st.session_state.username.upper()}</div>',
        unsafe_allow_html=True)

    # ── DB persistence status badge ────────────────────────────────────────────
    if _DB_STATUS == "postgres":
        st.markdown(
            '<div style="font-size:.7rem;font-weight:700;color:#10b981;'
            'background:rgba(16,185,129,.1);border:1px solid rgba(16,185,129,.3);'
            'border-radius:6px;padding:.3rem .6rem;margin-bottom:.8rem">'
            '🟢 Postgres connected — data persists</div>',
            unsafe_allow_html=True)
    elif _DB_STATUS == "sqlite_fallback":
        st.markdown(
            '<div style="font-size:.7rem;font-weight:700;color:#ef4444;'
            'background:rgba(239,68,68,.1);border:1px solid rgba(239,68,68,.3);'
            'border-radius:6px;padding:.3rem .6rem;margin-bottom:.8rem">'
            '🔴 Postgres failed — using temporary storage. '
            'Check DATABASE_URL secret.</div>',
            unsafe_allow_html=True)
        if _DB_ERROR:
            with st.expander("⚠️ DB error detail"):
                st.code(_DB_ERROR[:300])
    else:
        st.markdown(
            '<div style="font-size:.7rem;font-weight:700;color:#f59e0b;'
            'background:rgba(245,158,11,.1);border:1px solid rgba(245,158,11,.3);'
            'border-radius:6px;padding:.3rem .6rem;margin-bottom:.8rem">'
            '🟡 Local storage (data resets on restart). '
            'Add DATABASE_URL to persist.</div>',
            unsafe_allow_html=True)

    if st.button("🚪 Logout", width="stretch"):
        controller.set("swing_user_id", "", max_age=0)
        st.session_state.clear()
        st.rerun()

    st.markdown("<hr style='margin:1rem 0;border-color:var(--border)'>",
                unsafe_allow_html=True)
    st.markdown('<div style="font-size:.8rem;font-weight:800;letter-spacing:.05em">'
                '🎨 UI THEME</div>', unsafe_allow_html=True)

    new_theme = st.selectbox(
        "Theme", list(THEMES.keys()),
        index=list(THEMES.keys()).index(st.session_state.theme),
        label_visibility="collapsed")
    if new_theme != st.session_state.theme:
        st.session_state.theme = new_theme
        st.rerun()

    st.markdown("<hr style='margin:.8rem 0;border-color:var(--border)'>",
                unsafe_allow_html=True)
    st.markdown('<div style="font-size:.8rem;font-weight:800;letter-spacing:.05em;'
                'margin-bottom:.5rem">🗺 NAVIGATION</div>', unsafe_allow_html=True)

    NAV_GROUPS = {
        "📊 Portfolio": [
            ("📋 Overview",          "portfolio"),
            ("📊 Charts",            "analytics"),
            ("📈 Stock Chart",       "chart"),
            ("📐 Metrics",           "metrics"),
            ("📤 Export",            "export"),
        ],
        "🔔 Signals & Alerts": [
            ("🔔 Active Signals",    "signals"),
            ("🪤 Trap Scanner",      "traps"),
            ("🏦 Smart Money (SMC)", "smc"),
            ("📐 VCP Scanner",       "vcp"),
        ],
        "🔄 Market Intelligence": [
            ("🔄 Sector Rotation",   "sector"),
            ("💪 RS Leaders",        "rs"),
            ("🌌 Universe Scanner",  "scanner"),
            ("📊 Market Breadth",    "breadth"),
            ("🔬 Custom Screener",   "screener"),
            ("📅 Corporate Actions", "corp_actions"),
            ("📆 Earnings Calendar", "earnings"),
            ("🆕 IPO Tracker",       "ipo"),
        ],
        "🛡 Risk & Sizing": [
            ("🧮 Position Sizing",   "sizing"),
            ("🛡 Risk Dashboard",    "risk"),
        ],
        "🛠 Tools": [
            ("👁 Watchlist",         "watchlist"),
            ("🔔 Price Alerts",      "alerts"),
            ("📓 Trade Journal",     "journal"),
            ("🎯 Signal Scores",     "scores"),
        ],
    }
    # Flat list for radio
    nav_labels = [label for group in NAV_GROUPS.values() for label, _ in group]
    nav_keys   = [key   for group in NAV_GROUPS.values() for _, key   in group]

    if "active_page" not in st.session_state:
        st.session_state.active_page = "portfolio"

    # Group headers + radio buttons styled with CSS
    nav_html = ""
    flat_idx = 0
    for group_label, items in NAV_GROUPS.items():
        nav_html += (f'<div style="font-size:.65rem;font-weight:800;color:var(--muted);'
                     f'text-transform:uppercase;letter-spacing:.1em;margin:.6rem 0 .2rem;'
                     f'padding-left:.3rem">{group_label}</div>')
        flat_idx += len(items)

    # Use radio for actual selection (CSS handles grouping visually)
    cur_idx = nav_keys.index(st.session_state.active_page) \
              if st.session_state.active_page in nav_keys else 0
    sel_nav = st.radio(
        "nav", nav_labels, index=cur_idx,
        label_visibility="collapsed")
    new_page = nav_keys[nav_labels.index(sel_nav)]
    if new_page != st.session_state.active_page:
        st.session_state.active_page = new_page
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div style="font-size:1.1rem;font-weight:800;color:var(--accent);'
                'margin-bottom:.8rem">⚡ Trade Entry</div>', unsafe_allow_html=True)

    em   = st.session_state.edit_id is not None
    erow = (raw[raw["id"] == st.session_state.edit_id].iloc[0]
            if em and not raw.empty else None)
    if em:
        st.markdown(
            '<div style="background:rgba(99,102,241,.15);border:1px solid rgba(99,102,241,.4);'
            'border-radius:8px;padding:.5rem;font-size:.8rem;color:var(--accent);'
            'margin-bottom:1rem;font-weight:700">✏️ Editing trade</div>',
            unsafe_allow_html=True)

    with st.form("trade_form", clear_on_submit=True):
        s_in   = st.text_input("Stock Symbol",
                               value=erow["stock"] if erow is not None else "",
                               placeholder="CDSL, IRFC…")
        q_in   = st.number_input("Quantity", min_value=1, step=1,
                                 value=int(erow["quantity"]) if erow is not None else 1)
        b_in   = st.number_input("Buy At $", min_value=0.01, step=0.05,
                                 value=float(erow["buy_at"]) if erow is not None else 0.01,
                                 format="%.2f")
        sel_in = st.number_input(
            "Sell At $ (optional)", min_value=0.0, step=0.05,
            value=float(erow["sell_at"]) if (erow is not None and erow["sell_at"]) else 0.0,
            format="%.2f")

        if st.form_submit_button(
                "💾 Update Trade" if em else "➕ Execute Entry", width="stretch"):
            if not s_in.strip():
                st.error("Symbol required")
            elif b_in <= 0:
                st.error("Buy price must be > 0")
            else:
                sv = sel_in if sel_in > 0 else None
                if em:
                    update_trade(st.session_state.edit_id, UID, s_in, q_in, b_in,
                                 sv, "Closed" if sv else "Open")
                    st.session_state.edit_id = None
                    st.success("Updated!")
                else:
                    add_trade(UID, s_in, q_in, b_in, sv)
                    st.success(f"Added {s_in.upper()}")
                _CACHE.clear()
                st.session_state.last_auto_scan = 0.0
                st.rerun()

    if em and st.button("✖ Cancel Edit", width="stretch"):
        st.session_state.edit_id = None
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div style="font-size:.8rem;font-weight:800;letter-spacing:.05em">'
                '🔍 FILTERS</div>', unsafe_allow_html=True)
    st.session_state.filter_status = st.selectbox(
        "Status", ["All", "Open", "Closed"],
        index=["All", "Open", "Closed"].index(st.session_state.filter_status),
        label_visibility="collapsed")
    st.session_state.filter_pnl = st.selectbox(
        "P&L", ["All", "Profitable", "Loss"],
        index=["All", "Profitable", "Loss"].index(st.session_state.filter_pnl),
        label_visibility="collapsed")
    st.session_state.search = st.text_input(
        "Search", value=st.session_state.search,
        placeholder="Search symbol…", label_visibility="collapsed")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div style="font-size:.8rem;font-weight:800;letter-spacing:.05em">'
                '⚙️ SCAN CONTROLS</div>', unsafe_allow_html=True)

    _interval_opts = {"5 min": 300, "15 min": 900, "30 min": 1800}
    _interval_labels = list(_interval_opts.keys())

    # Core (fast) scan controls
    st.markdown('<div style="font-size:.72rem;color:var(--muted);font-weight:700;'
                'margin:.5rem 0 .2rem">⚡ Core scan (signals · news · prices)</div>',
                unsafe_allow_html=True)
    fc1, fc2 = st.columns([1, 1])
    with fc1:
        st.session_state.auto_fast = st.toggle(
            "Auto", value=st.session_state.auto_fast, key="toggle_fast")
    with fc2:
        _cur_fast = next((k for k, v in _interval_opts.items()
                          if v == st.session_state.fast_interval_sec), "5 min")
        _sel_fast = st.selectbox("Every", _interval_labels,
                                 index=_interval_labels.index(_cur_fast),
                                 key="sel_fast", label_visibility="collapsed")
        st.session_state.fast_interval_sec = _interval_opts[_sel_fast]
    if st.button("⚡ Scan Core Now", width="stretch"):
        st.session_state._manual_fast_request = True
        _cached_prices.clear()
        st.rerun()

    # Deep scan controls
    st.markdown('<div style="font-size:.72rem;color:var(--muted);font-weight:700;'
                'margin:.7rem 0 .2rem">🔄 Deep scan (sector · universe · SMC · trap · VCP · RS)</div>',
                unsafe_allow_html=True)
    dc1, dc2 = st.columns([1, 1])
    with dc1:
        st.session_state.auto_deep = st.toggle(
            "Auto", value=st.session_state.auto_deep, key="toggle_deep")
    with dc2:
        _cur_deep = next((k for k, v in _interval_opts.items()
                          if v == st.session_state.deep_interval_sec), "15 min")
        _sel_deep = st.selectbox("Every", _interval_labels,
                                 index=_interval_labels.index(_cur_deep),
                                 key="sel_deep", label_visibility="collapsed")
        st.session_state.deep_interval_sec = _interval_opts[_sel_deep]
    if st.button("🔄 Scan Deep Now", width="stretch"):
        st.session_state._manual_deep_request = True
        st.rerun()

    # Status / countdown
    _elapsed_fast = time.time() - st.session_state.last_auto_scan
    _elapsed_slow = time.time() - st.session_state.last_slow_scan
    _nxt_fast = max(0, int((st.session_state.fast_interval_sec - _elapsed_fast) // 60))
    _nxt_slow = max(0, int((st.session_state.deep_interval_sec - _elapsed_slow) // 60))
    _stage_names = {"sector": "Sector rotation", "universe": "Universe scan",
                    "smc": "SMC setups", "traps": "Trap scan",
                    "vcp": "VCP bases", "rs": "RS leaders"}
    if st.session_state.get("_deep_running", False):
        _cur = st.session_state.get("_deep_stage", "sector")
        _deep_status = f'⏳ {_stage_names.get(_cur, _cur)}…'
    else:
        _deep_status = f'{_nxt_slow}m' if st.session_state.auto_deep else 'manual'
    _fast_status = f'{_nxt_fast}m' if st.session_state.auto_fast else 'manual'
    st.markdown(
        f'<div style="font-size:.68rem;color:var(--muted);padding-top:.5rem;'
        f'font-weight:600;line-height:1.6">'
        f'⚡ core: {_fast_status} · 🔄 deep: {_deep_status}</div>',
        unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div style="font-size:.8rem;font-weight:800;letter-spacing:.05em">'
                '📱 TELEGRAM</div>', unsafe_allow_html=True)

    # Load from DB first; fall back to st.secrets; cache in session_state
    # so values survive within-session navigation without a DB re-read.
    if "tg_tok_saved" not in st.session_state or "tg_cid_saved" not in st.session_state:
        db_tok, db_cid = get_tg_config(UID)
        if not db_tok:
            try:
                db_tok = st.secrets.get("telegram_bot_token", "")
                db_cid = st.secrets.get("telegram_chat_id", "")
            except Exception:
                db_tok = db_cid = ""
        st.session_state.tg_tok_saved = db_tok or ""
        st.session_state.tg_cid_saved = db_cid or ""

    saved_tok = st.session_state.tg_tok_saved
    saved_cid = st.session_state.tg_cid_saved

    tg_tok = st.text_input("Bot Token", value=saved_tok, type="password")
    tg_cid = st.text_input("Chat ID",   value=saved_cid)
    if st.button("💾 Save Config", width="stretch"):
        save_tg_config(UID, tg_tok, tg_cid)
        st.session_state.tg_tok_saved = tg_tok
        st.session_state.tg_cid_saved = tg_cid
        st.success("Saved!")

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="dash-title">'
    '<div class="dash-title-text">📈 Quantitative <span class="hl">Swing Dashboard</span></div>'
    '<span class="refresh-badge">⚡ SIGNALS LIVE · 🔄 SECTOR LIVE</span>'
    '</div>',
    unsafe_allow_html=True)

market = _get_market_regime_safe()
regime = market.get("regime", "Unknown")

rc_map = {
    "Strong Bull":  ("rgba(16,185,129,.15)",  "#10b981", "border:1px solid rgba(16,185,129,.4)"),
    "Bull":         ("rgba(16,185,129,.1)",   "#10b981", "border:1px solid rgba(16,185,129,.2)"),
    "Bull Pullback":("rgba(245,158,11,.15)",  "#f59e0b", "border:1px solid rgba(245,158,11,.4)"),
    "Strong Bear":  ("rgba(239,68,68,.15)",   "#ef4444", "border:1px solid rgba(239,68,68,.4)"),
    "Bear":         ("rgba(239,68,68,.1)",    "#ef4444", "border:1px solid rgba(239,68,68,.2)"),
    "Bear Rally":   ("rgba(245,158,11,.15)",  "#f59e0b", "border:1px solid rgba(245,158,11,.4)"),
}
rc_bg, rc_clr, rc_border = rc_map.get(
    regime, ("rgba(148,163,184,.1)", "#94a3b8", "border:1px solid rgba(148,163,184,.3)"))

indices_html = ""
_idx_items = market.get("indices", {})
for name, d in _idx_items.items():
    price = d.get("price")
    chg   = d.get("chg_pct", 0)
    price_str = f"${price:,.0f}" if price else "—"
    if name == "VIX":
        chg_clr = "var(--red)" if chg > 0 else "var(--green)"
    else:
        chg_clr = "var(--green)" if chg > 0 else "var(--red)"
    indices_html += (
        f'<span style="color:var(--text);font-size:.8rem;padding:0 .8rem;'
        f'border-right:1px solid rgba(255,255,255,.1)">'
        f'{name} <b>{price_str}</b> '
        f'<span style="color:{chg_clr};font-weight:700">{chg:+.2f}%</span></span>'
    )
if not _idx_items:
    # Indices failed to load — show a clear note instead of a blank banner
    indices_html = (
        '<span style="color:var(--muted);font-size:.78rem;padding:0 .8rem">'
        '📡 Index data loading… (Yahoo Finance may be rate-limited; '
        'refreshes automatically)</span>'
    )

sup_str = f"${market.get('support'):,.0f}" if market.get("support") else "—"
res_str = f"${market.get('resistance'):,.0f}" if market.get("resistance") else "—"

st.markdown(
    f'<div class="regime-banner" style="background:{rc_bg};{rc_border};'
    f'backdrop-filter:blur(10px)">'
    f'<span style="color:{rc_clr};font-weight:800;font-size:.9rem;'
    f'white-space:nowrap;letter-spacing:.05em">'
    f'🌐 {regime.upper()} (CONF: {market.get("confidence","—")}%)</span>'
    f'{indices_html}'
    f'<span style="color:var(--muted);font-size:.75rem;white-space:nowrap;'
    f'padding-left:.5rem;font-weight:600">'
    f'SUP: {sup_str} | RES: {res_str} | '
    f'RSI {market.get("nifty_rsi","—")} | RISK: {market.get("risk_level","—")}'
    f'</span></div>',
    unsafe_allow_html=True)

# ── KPI cards ──────────────────────────────────────────────────────────────────
pnl_c = "green" if t_pnl >= 0 else "red"
r_c   = "green" if t_real >= 0 else "red"
u_c   = "green" if t_unreal >= 0 else "red"

st.markdown(
    '<div class="cards">'
    + card("Total Invested",  fi(t_inv),    "",          "blue")
    + card("Portfolio Value", fi(t_cur),    "",          "blue")
    + card("Total P&L",       fi(t_pnl),    fp(t_pnl_pct), pnl_c)
    + card("Realized P&L",    fi(t_real),   "",          r_c)
    + card("Unrealized P&L",  fi(t_unreal), "",          u_c)
    + card("Open Trades",     str(len(odf)), "Active",   "yellow")
    + card("Closed Trades",   str(len(cdf)), "Historical","green" if len(cdf) > 0 else "")
    + card("Best Trade 🏆",   best,          "",         "green")
    + card("Worst Trade 📉",  worst,         "",         "red")
    + '</div>',
    unsafe_allow_html=True)

# ── Tabs ────────────────────────────────────────────────────────────────────────
# ── Page routing — driven by sidebar navigation ────────────────────────────────
_page = st.session_state.get("active_page", "portfolio")

# ── Portfolio ────────────────────────────────────────────────────────────────
if _page == 'portfolio':
    if df.empty:
        st.info("No trades yet. Use the sidebar to execute an entry.")
    else:
        fdf = df.copy()
        if st.session_state.filter_status != "All":
            fdf = fdf[fdf["status"] == st.session_state.filter_status]
        if st.session_state.filter_pnl == "Profitable":
            fdf = fdf[fdf["profit"] > 0]
        elif st.session_state.filter_pnl == "Loss":
            fdf = fdf[fdf["profit"] < 0]
        if st.session_state.search.strip():
            fdf = fdf[fdf["stock"].str.upper().str.contains(
                st.session_state.search.upper())]

        sort_opts = {
            "Stock": "stock", "Qty": "quantity", "Buy At": "buy_at",
            "CMP": "cmp", "Invested": "invested",
            "P&L $": "profit", "P&L %": "profit_pct"
        }
        sc1, sc2 = st.columns([3, 1])
        with sc1:
            sort_key = st.selectbox(
                "Sort by", list(sort_opts.keys()),
                index=list(sort_opts.values()).index(st.session_state.sort_col)
                if st.session_state.sort_col in sort_opts.values() else 0,
                label_visibility="collapsed")
            sort_col = sort_opts[sort_key]
        with sc2:
            asc = st.toggle("⬆ Ascending", value=st.session_state.sort_asc)
            st.session_state.sort_asc = asc

        st.session_state.sort_col = sort_col
        if sort_col in fdf.columns:
            fdf = fdf.sort_values(sort_col, ascending=asc, na_position="last")

        st.markdown(
            f'<div class="sec">Open Positions & History ({len(fdf)})</div>',
            unsafe_allow_html=True)

        rows_html = ""
        for _, r in fdf.iterrows():
            row_cls = ("row-profit" if r.get("profit", 0) > 0
                       else "row-loss" if r.get("profit", 0) < 0 else "row-neutral")
            cmp_cell = (
                '<td class="zero-cell">—</td>' if pd.isna(r.get("cmp"))
                else f'<td class="pos">{fi2(r["cmp"])}</td>' if r["cmp"] > r["buy_at"]
                else f'<td class="neg">{fi2(r["cmp"])}</td>' if r["cmp"] < r["buy_at"]
                else f'<td>{fi2(r["cmp"])}</td>'
            )
            cur_cell = (
                '<td>—</td>' if pd.isna(r.get("current_amt", 0))
                else f'<td class="pos">{fi(r["current_amt"])}</td>'
                     if r["current_amt"] > r["invested"]
                else f'<td class="neg">{fi(r["current_amt"])}</td>'
                     if r["current_amt"] < r["invested"]
                else f'<td>{fi(r["current_amt"])}</td>'
            )
            rows_html += (
                f"<tr class='{row_cls}'>"
                f"<td class='l'><span class='nse-lbl'>{r.get('nse_label','')}</span></td>"
                f"<td class='l'><b style='font-size:.9rem'>{r['stock']}</b><br>"
                f"<span style='font-size:.7rem;color:var(--muted)'>"
                f"{get_sector(r['stock'])} · {r.get('added_date','')}</span></td>"
                f"<td>{int(r['quantity'])}</td>"
                f"<td>{fi2(r['buy_at'])}</td>"
                f"{cmp_cell}"
                f"<td>{'—' if pd.isna(r.get('sell_at')) else fi2(r['sell_at'])}</td>"
                f"<td>{fi(r['invested'])}</td>"
                f"{cur_cell}"
                f"{cv_cell(r.get('profit', 0), fi)}"
                f"{cv_cell(r.get('profit_pct', 0), fp)}"
                f"<td>{badge(r['status'], r.get('profit', 0))}</td>"
                f"</tr>"
            )

        st.markdown(
            f'<div class="tbl-wrap"><table class="t"><thead><tr>'
            f'<th class="l">NSE</th><th class="l">Asset</th>'
            f'<th>Qty</th><th>Entry</th><th>CMP</th><th>Exit</th>'
            f'<th>Invested</th><th>Value</th><th>P&L $</th><th>P&L %</th>'
            f'<th>Status</th></tr></thead><tbody>{rows_html}</tbody></table></div>',
            unsafe_allow_html=True)

        st.markdown('<div class="sec">Manage Positions</div>', unsafe_allow_html=True)
        opts = [f"{r['id']} — {r['stock']}" for _, r in fdf.iterrows()]
        if opts:
            ca, cb, cc, cd = st.columns([3, 1, 1, 1])
            with ca:
                sel_id = int(
                    st.selectbox("Select Trade ID", opts,
                                 label_visibility="collapsed").split(" — ")[0])
            with cb:
                if st.button("✏️ Modify", width="stretch"):
                    st.session_state.edit_id = sel_id
                    st.rerun()
            with cc:
                if st.button("🔒 Close Pos", width="stretch"):
                    st.session_state.close_id = sel_id
                    st.rerun()
            with cd:
                if st.button("🗑 Drop", width="stretch"):
                    st.session_state.del_id = sel_id
                    st.rerun()

        if st.session_state.close_id:
            st.markdown("---")
            st.markdown("**Execute Close — Confirm Exit Price**")
            sp = st.number_input("Exit Price $", min_value=0.01, step=0.05, format="%.2f")
            x1, x2 = st.columns(2)
            with x1:
                if st.button("✅ Confirm Exit", width="stretch"):
                    close_trade(st.session_state.close_id, UID, sp)
                    st.session_state.close_id = None
                    st.rerun()
            with x2:
                if st.button("✖ Abort", width="stretch"):
                    st.session_state.close_id = None
                    st.rerun()

        if st.session_state.del_id:
            st.markdown("---")
            st.warning(
                f"Drop trade ID #{st.session_state.del_id}? This is irreversible.")
            y1, y2 = st.columns(2)
            with y1:
                if st.button("🗑 Confirm Drop", width="stretch"):
                    delete_trade(st.session_state.del_id, UID)
                    st.session_state.del_id = None
                    st.rerun()
            with y2:
                if st.button("✖ Abort", width="stretch"):
                    st.session_state.del_id = None
                    st.rerun()

# ── Charts / Analytics ───────────────────────────────────────────────────────
elif _page == 'analytics':
    if df.empty:
        st.info("Execute trades to populate visualization models.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(chart_alloc(df), use_container_width=True)
        with c2:
            st.plotly_chart(chart_donut(df), use_container_width=True)
        st.plotly_chart(chart_pnl(df), use_container_width=True)
        st.plotly_chart(
            chart_growth(get_history(UID), t_cur, t_inv),
            use_container_width=True)

        # ── Enhanced analytics (Batch 2) ──────────────────────────────────────
        st.markdown("<hr style='border-color:var(--border);margin:1.5rem 0'>",
                    unsafe_allow_html=True)
        st.markdown('<div class="sec">📊 Performance Breakdown</div>',
                    unsafe_allow_html=True)

        closed = df[df["status"] == "Closed"].copy()

        ec1, ec2 = st.columns(2)

        # 1. P&L by sector (realized, closed trades)
        with ec1:
            if not closed.empty:
                closed["sector"] = closed["stock"].apply(get_sector)
                sec_pnl = closed.groupby("sector")["profit"].sum().sort_values()
                colors = ["#ef4444" if v < 0 else "#10b981" for v in sec_pnl.values]
                bfig = go.Figure(go.Bar(
                    x=sec_pnl.values, y=sec_pnl.index, orientation="h",
                    marker_color=colors))
                bfig.update_layout(
                    title="Realized P&L by Sector", height=300,
                    margin=dict(l=10, r=10, t=40, b=10),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color=theme_t.get("text", "#fff")))
                bfig.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
                st.plotly_chart(bfig, use_container_width=True)
            else:
                st.info("No closed trades yet for sector P&L.")

        # 2. Win/Loss distribution
        with ec2:
            if not closed.empty:
                wins = len(closed[closed["profit"] > 0])
                losses = len(closed[closed["profit"] < 0])
                be = len(closed[closed["profit"] == 0])
                pfig = go.Figure(go.Pie(
                    labels=["Wins", "Losses", "Breakeven"],
                    values=[wins, losses, be], hole=0.5,
                    marker_colors=["#10b981", "#ef4444", "#f59e0b"]))
                pfig.update_layout(
                    title="Win / Loss Distribution", height=300,
                    margin=dict(l=10, r=10, t=40, b=10),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color=theme_t.get("text", "#fff")))
                st.plotly_chart(pfig, use_container_width=True)
            else:
                st.info("No closed trades yet for win/loss split.")

        # 3. Best & worst trades
        if not closed.empty:
            st.markdown("##### 🏆 Best & Worst Closed Trades")
            closed_sorted = closed.sort_values("profit_pct", ascending=False)
            bw1, bw2 = st.columns(2)
            with bw1:
                st.markdown("**🟢 Top 5 Winners**")
                top5 = closed_sorted.head(5)[["stock", "profit", "profit_pct"]].copy()
                top5.columns = ["Stock", "P&L $", "P&L %"]
                st.dataframe(top5, width="stretch", hide_index=True)
            with bw2:
                st.markdown("**🔴 Top 5 Losers**")
                bot5 = closed_sorted.tail(5)[["stock", "profit", "profit_pct"]].copy()
                bot5 = bot5.sort_values("P&L %" if "P&L %" in bot5.columns else "profit_pct")
                bot5.columns = ["Stock", "P&L $", "P&L %"]
                st.dataframe(bot5, width="stretch", hide_index=True)

        # 4. Holding period distribution
        if not closed.empty and "closed_date" in closed.columns:
            try:
                closed["hold_days"] = (
                    pd.to_datetime(closed["closed_date"]) -
                    pd.to_datetime(closed["added_date"])).dt.days
                valid_hold = closed["hold_days"].dropna()
                if not valid_hold.empty:
                    st.markdown("##### ⏱ Holding Period Distribution")
                    hfig = go.Figure(go.Histogram(
                        x=valid_hold, marker_color="#3b82f6", nbinsx=20))
                    hfig.update_layout(
                        height=260, margin=dict(l=10, r=10, t=10, b=10),
                        xaxis_title="Days held", yaxis_title="Trades",
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color=theme_t.get("text", "#fff")))
                    hfig.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
                    hfig.update_yaxes(gridcolor="rgba(255,255,255,0.05)")
                    st.plotly_chart(hfig, use_container_width=True)
                    avg_hold = valid_hold.mean()
                    st.caption(f"Average holding period: {avg_hold:.0f} days")
            except Exception:
                pass

# ── Active Signals ───────────────────────────────────────────────────────────
elif _page == 'signals':
    st.markdown(
        '<div class="sec">Active Portfolio Signals & Risk Management</div>',
        unsafe_allow_html=True)

    s1, s2 = st.columns([2, 1])
    with s1:
        st.caption("🤖 Neural background scan refreshes every 15 minutes.")
    with s2:
        if st.button("📲 Push to Telegram", width="stretch",
                     disabled=not bool(saved_tok and saved_cid)):
            if st.session_state.signals_cache is not None:
                with st.spinner("🤖 Compiling Telegram report..."):
                    msg_payload = build_telegram_message(
                        st.session_state.signals_cache,
                        st.session_state.sector_cache
                        if st.session_state.sector_cache is not None else pd.DataFrame(),
                        st.session_state.picks_cache
                        if st.session_state.picks_cache is not None else []
                    )
                    news = st.session_state.news_cache or []
                    if news:
                        msg_payload += "\n\n🌍 <b>LATEST HOLDINGS NEWS</b>\n"
                        msg_payload += "\n".join(news[:8])
                    ok = send_telegram(saved_tok, saved_cid, msg_payload)
                    if ok:
                        st.success("✅ Broadcast successful!")
                    else:
                        st.error("❌ Broadcast failed. Check token/chat ID.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec">🌍 Live Portfolio News</div>',
                unsafe_allow_html=True)

    with st.expander("📰 Latest Headlines for Active Holdings", expanded=False):
        open_raw = raw[raw["status"] == "Open"] if not raw.empty else pd.DataFrame()
        if not open_raw.empty:
            col_news1, col_news2 = st.columns([3, 1])
            with col_news2:
                force_news = st.button("🔄 Refresh News", width="stretch")
            if force_news:
                with st.spinner("Fetching latest headlines..."):
                    st.session_state.news_cache = fetch_portfolio_news(open_raw)

            render_news(st.session_state.news_cache or [])
        else:
            st.info("No active trades. Add a trade to see related news.")

    if st.session_state.signals_cache is not None:
        nc = {"SELL": 0, "AVERAGE": 0, "HOLD": 0, "WATCH": 0}
        for s in st.session_state.signals_cache:
            for k in nc:
                if k in s.get("action", ""):
                    nc[k] += 1

        st.markdown(
            f'<div style="display:flex;gap:.8rem;margin:.5rem 0 1rem">'
            f'<span style="background:rgba(239,68,68,.15);color:#ef4444;'
            f'padding:.3rem .8rem;border-radius:6px;font-size:.8rem;font-weight:800;'
            f'border:1px solid rgba(239,68,68,.3)">🔴 SELL: {nc["SELL"]}</span>'
            f'<span style="background:rgba(245,158,11,.15);color:#f59e0b;'
            f'padding:.3rem .8rem;border-radius:6px;font-size:.8rem;font-weight:800;'
            f'border:1px solid rgba(245,158,11,.3)">🟡 AVERAGE: {nc["AVERAGE"]}</span>'
            f'<span style="background:rgba(16,185,129,.15);color:#10b981;'
            f'padding:.3rem .8rem;border-radius:6px;font-size:.8rem;font-weight:800;'
            f'border:1px solid rgba(16,185,129,.3)">🟢 HOLD: {nc["HOLD"]}</span>'
            f'<span style="background:rgba(148,163,184,.1);color:#94a3b8;'
            f'padding:.3rem .8rem;border-radius:6px;font-size:.8rem;font-weight:800;'
            f'border:1px solid rgba(148,163,184,.3)">⚪ WATCH: {nc["WATCH"]}</span>'
            f'</div>',
            unsafe_allow_html=True)

        render_signals(st.session_state.signals_cache, theme_t)

# ── Sector Rotation ──────────────────────────────────────────────────────────
elif _page == 'sector':
    st.markdown('<div class="sec">Macro Sector Rotation & Capital Flow</div>',
                unsafe_allow_html=True)

    if st.session_state.sector_cache is not None:
        render_sector(st.session_state.sector_cache, theme_t)

        if not st.session_state.sector_cache.empty:
            top = st.session_state.sector_cache.iloc[0]
            rs_val  = top.get("rs_vs_nifty_1m", 0) or 0
            rs_clr  = "#10b981" if rs_val > 0 else "#ef4444"
            rrg_val = top.get("rrg_quadrant", "—")

            st.markdown(
                f'<div style="margin-top:1rem;background:rgba(16,185,129,.08);'
                f'border:1px solid rgba(16,185,129,.3);border-radius:8px;'
                f'padding:.8rem 1rem;font-size:.85rem">'
                f'🥇 <b style="color:var(--text)">Leading Sector: {top["sector"]}</b>'
                f' — Momentum {top["momentum_score"]:.2f}'
                f' | Avg RSI {top["avg_rsi"]:.0f}'
                f' | Flow {top["avg_pct"]:+.1f}%'
                f' | RS vs S&P <b style="color:{rs_clr}">{rs_val:+.1f}%</b>'
                f' | {rrg_val}<br>'
                f'<span style="color:var(--muted);font-size:.75rem;margin-top:5px;'
                f'display:block">Constituents: {top["stocks"]}</span>'
                f'</div>',
                unsafe_allow_html=True)

    if (st.session_state.outlook_cache is not None and
            not st.session_state.outlook_cache.empty):
        st.markdown(
            '<div class="sec" style="margin-top:2rem">📈 Institutional Outlook</div>',
            unsafe_allow_html=True)
        render_outlook(st.session_state.outlook_cache, theme_t)

    if st.session_state.picks_cache is not None:
        st.markdown(
            '<div class="sec" style="margin-top:2rem">🎯 Algorithmic Entry Setups</div>',
            unsafe_allow_html=True)
        render_picks(st.session_state.picks_cache, theme_t)

# ── Universe Scanner ─────────────────────────────────────────────────────────
elif _page == 'scanner':
    st.markdown(
        f'<div class="sec">🌌 Universe Scanner — top {min(MAX_SCAN_SYMBOLS, UNIVERSE_TOTAL):,} liquid of {UNIVERSE_TOTAL:,}</div>',
        unsafe_allow_html=True)

    # ── Universe source breakdown ─────────────────────────────────────────────
    src_html = ""
    for lbl, n, sk, err in UNIVERSE_SOURCES:
        clr = theme_t["green"] if n > 0 else theme_t["red"]
        src_html += (
            f'<span style="background:var(--card2);border:1px solid var(--border);'
            f'border-radius:6px;padding:.25rem .6rem;font-size:.72rem;'
            f'font-weight:700;color:var(--text)">'
            f'{"📄" if n > 0 else "❌"} {lbl} '
            f'<span style="color:{clr}">{n:,}</span></span> '
        )
    st.markdown(
        f'<div style="display:flex;gap:.4rem;flex-wrap:wrap;margin-bottom:.8rem;'
        f'align-items:center">'
        f'<span style="font-size:.75rem;color:var(--muted);font-weight:600">'
        f'Sources loaded:</span> {src_html}</div>',
        unsafe_allow_html=True)

    # ── Diagnostic expander — shows exactly what loaded and why ───────────────
    with st.expander("🔍 Universe Load Diagnostics", expanded=False):
        st.code(debug_universe_load(), language=None)
        st.caption("If a file shows '❌ not found', check it is committed to "
                   "your repo root (same folder as signals.py and app.py).")

    st.markdown(
        f'<div style="background:rgba(212,175,55,.06);border:1px solid rgba(212,175,55,.2);'
        f'border-radius:8px;padding:.7rem 1rem;font-size:.8rem;color:var(--muted);'
        f'margin-bottom:1rem;line-height:1.6">'
        f'💡 <b style="color:var(--text)">Universe:</b> Edit '
        f'<code>us_universe.csv</code> (Symbol,Name,Sector) and commit to your repo root.<br>'
        f'Run <code>python build_us_universe.py</code> to refresh the liquid-filtered '
        f'<code>us_universe_liquid.csv</code> (price + average dollar-volume screen). '
        f'ETFs live in <code>us_etfs.csv</code>.'
        f'</div>',
        unsafe_allow_html=True)

    # ── Custom stock input ─────────────────────────────────────────────────────
    with st.expander("➕ Add Custom Stocks to Scan", expanded=False):
        st.caption("Enter NSE symbols (comma-separated). These are added to the scan universe temporarily.")
        custom_raw = st.text_area(
            "Custom symbols", value=st.session_state.get("custom_stocks_input",""),
            placeholder="IRFC, CDSL, SNOWMAN, ZOMATO...",
            label_visibility="collapsed", height=80)
        if st.button("✅ Apply Custom List"):
            # Store and inject into signals module
            symbols = [s.strip().upper() for s in custom_raw.split(",") if s.strip()]
            st.session_state.custom_stocks_input = custom_raw
            if symbols:
                import signals as _sg
                if "Custom" not in _sg.SECTOR_STOCKS:
                    _sg.SECTOR_STOCKS["Custom"] = []
                _sg.SECTOR_STOCKS["Custom"] = symbols
                for sym in symbols:
                    _sg.SECTOR_MAP[sym] = "Custom"
                st.success(f"✅ Added {len(symbols)} custom stocks to scan universe")
            else:
                import signals as _sg
                _sg.SECTOR_STOCKS.pop("Custom", None)
                st.info("Custom list cleared.")

    if st.button("⚡ Execute Global Scan", width="stretch"):
        with st.spinner(f"Scanning {min(MAX_SCAN_SYMBOLS, len(SECTOR_MAP))} tickers..."):
            sd = generate_market_scanner()
            st.session_state.scanner_cache = sd if (sd is not None and not sd.empty) \
                else pd.DataFrame()
            if (st.session_state.scanner_cache is not None and
                    not st.session_state.scanner_cache.empty):
                _sc = st.session_state.scanner_cache
                _has_liq = "Liquid" in _sc.columns
                liq_ok  = int((_sc["Liquid"]=="✅").sum()) if _has_liq else 0
                liq_low = int((_sc["Liquid"]=="⚠️ Low").sum()) if _has_liq else 0
                st.toast(
                    f"✅ {len(_sc)} setups | "
                    f"💧 {liq_ok} liquid · ⚠️ {liq_low} low-liq",
                    icon="🚀")
            st.rerun()

    scan_df = st.session_state.scanner_cache
    if scan_df is None:
        st.info("💡 Initiate scan above or await automated background scan.")
    elif scan_df.empty:
        st.warning("⚠️ Zero setups passed pattern gates today.")
    else:
        all_sectors = sorted(scan_df["Sector"].unique().tolist())

        # ── Sector summary cards ───────────────────────────────────────────────
        _has_liq_col = "Liquid" in scan_df.columns
        _agg = {"total": ("Stock","count"),
                "strong": ("Signal", lambda x: (x=="🔥 STRONG BUY").sum()),
                "buy":    ("Signal", lambda x: (x=="🟢 BUY SETUP").sum())}
        if _has_liq_col:
            _agg["liq"] = ("Liquid", lambda x: (x=="✅").sum())
        sector_stats = scan_df.groupby("Sector").agg(**_agg).reset_index()
        if not _has_liq_col:
            sector_stats["liq"] = 0
        cards_html = ""
        for _, sr in sector_stats.iterrows():
            hot = sr["strong"] + sr["buy"]
            clr = theme_t["green"] if hot > 0 else theme_t["muted"]
            bdr = theme_t["accent"] if hot > 0 else theme_t["border"]
            cards_html += (
                f'<div style="background:var(--card);border:1px solid {bdr};'
                f'border-radius:10px;padding:.8rem 1rem;min-width:150px;flex:1">'
                f'<div style="font-size:.78rem;font-weight:800;color:var(--text);'
                f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis">'
                f'{sr["Sector"]}</div>'
                f'<div style="font-size:1.1rem;font-weight:800;color:{clr};margin:.2rem 0">'
                f'{int(sr["total"])}<span style="font-size:.65rem;color:var(--muted)"> stocks</span></div>'
                f'<div style="font-size:.68rem;color:var(--muted)">'
                f'🔥{int(sr["strong"])} 🟢{int(sr["buy"])} '
                f'· 💧{int(sr["liq"])} liquid</div></div>'
            )
        st.markdown(
            f'<div style="display:flex;gap:.5rem;flex-wrap:wrap;'
            f'margin-bottom:1rem">{cards_html}</div>',
            unsafe_allow_html=True)

        # ── Stable filter controls ──────────────────────────────────────────────
        fc1, fc2, fc3, fc4, fc5 = st.columns([2, 1.5, 1.2, 1, 1])
        with fc1:
            sector_options = ["All Sectors"] + all_sectors
            sel_sector = st.selectbox("Sector", sector_options,
                index=sector_options.index(st.session_state.selected_scanner_sector)
                if st.session_state.selected_scanner_sector in sector_options else 0,
                label_visibility="collapsed")
            if sel_sector != st.session_state.selected_scanner_sector:
                st.session_state.selected_scanner_sector = sel_sector
        with fc2:
            signal_opts = ["All Signals","🔥 STRONG BUY","🟢 BUY SETUP",
                           "🟡 ACCUMULATE","⚪ NEUTRAL","🔴 AVOID"]
            sel_signal = st.selectbox("Signal", signal_opts, label_visibility="collapsed")
        with fc3:
            _has_liq_col2 = "Liquid" in scan_df.columns if scan_df is not None else False
            liq_opts = (["All","✅ Liquid Only","⚠️ Low Liq Only"]
                        if _has_liq_col2 else ["All"])
            sel_liq = st.selectbox("Liquidity", liq_opts, label_visibility="collapsed")
        with fc4:
            search_stock = st.text_input("Search", placeholder="Symbol",
                                         label_visibility="collapsed")
        with fc5:
            min_score_f = st.number_input("Min score", 0, 10, 0, 1,
                                          label_visibility="collapsed")

        # ── Apply filters ───────────────────────────────────────────────────────
        fdf = scan_df.copy()
        if sel_sector != "All Sectors":
            fdf = fdf[fdf["Sector"] == sel_sector]
        if sel_signal != "All Signals":
            fdf = fdf[fdf["Signal"] == sel_signal]
        if "Liquid" in fdf.columns:
            if sel_liq == "✅ Liquid Only":
                fdf = fdf[fdf["Liquid"] == "✅"]
            elif sel_liq == "⚠️ Low Liq Only":
                fdf = fdf[fdf["Liquid"] == "⚠️ Low"]
        if search_stock.strip():
            fdf = fdf[fdf["Stock"].str.upper().str.contains(
                search_stock.strip().upper())]
        if min_score_f > 0:
            fdf = fdf[fdf["Score"] >= min_score_f]

        # ── Strategy quick-filters (VCP / RS leaders / hide traps) ──────────────
        if any(col in fdf.columns for col in ("VCP", "RS", "Trap")):
            qf1, qf2, qf3, qf4 = st.columns(4)
            with qf1:
                only_vcp = st.checkbox("📐 VCP bases only", value=False, key="scn_vcp")
            with qf2:
                only_rs = st.checkbox("💪 RS leaders only", value=False, key="scn_rs")
            with qf3:
                ready_only = st.checkbox("🎯 VCP pivot-ready", value=False, key="scn_ready")
            with qf4:
                hide_traps = st.checkbox("🚫 Hide traps", value=False, key="scn_notrap")
            if only_vcp and "VCP" in fdf.columns:
                fdf = fdf[fdf["VCP"] != "—"]
            if ready_only and "VCP" in fdf.columns:
                fdf = fdf[fdf["VCP"].str.contains("READY", na=False)]
            if only_rs and "RS_Lead" in fdf.columns:
                fdf = fdf[fdf["RS_Lead"] == "💪"]
            if hide_traps and "Trap" in fdf.columns:
                fdf = fdf[fdf["Trap"] == "—"]

        display_df = fdf.drop(columns=["Sector"]) if sel_sector != "All Sectors" else fdf

        liq_count = int((fdf["Liquid"]=="✅").sum()) if "Liquid" in fdf.columns else 0
        st.markdown(
            f'<div style="font-size:.78rem;color:var(--muted);margin-bottom:.4rem;'
            f'font-weight:600">Showing {len(fdf)} of {len(scan_df)} results · '
            f'💧 {liq_count} liquid</div>',
            unsafe_allow_html=True)

        # ── Single stable dataframe with fixed height ───────────────────────────
        st.dataframe(
            display_df.reset_index(drop=True),
            hide_index=True, height=600, use_container_width=True,
            column_config={
                "Generated":    st.column_config.TextColumn("Time",     width="small"),
                "Sector":       st.column_config.TextColumn("Sector",   width="medium"),
                "Stock":        st.column_config.TextColumn("Stock",    width="small"),
                "Signal":       st.column_config.TextColumn("Signal",   width="medium"),
                "Liquid":       st.column_config.TextColumn("💧 Liq",   width="small"),
                "Turnover_M":  st.column_config.NumberColumn("$M/day",format="%.1f"),
                "Score":        st.column_config.NumberColumn("Score",  format="%d"),
                "CMP":     st.column_config.NumberColumn("CMP",    format="$%.2f"),
                "Entry":   st.column_config.NumberColumn("Entry",  format="$%.2f"),
                "Target":  st.column_config.NumberColumn("Target", format="$%.2f"),
                "SL":      st.column_config.NumberColumn("SL",     format="$%.2f"),
                "Support": st.column_config.NumberColumn("Support",format="$%.2f"),
                "Resist":  st.column_config.NumberColumn("Resist", format="$%.2f"),
                "RSI":     st.column_config.NumberColumn("RSI",    format="%.1f"),
                "Trend":   st.column_config.TextColumn("Trend",   width="medium"),
                "VCP":     st.column_config.TextColumn("📐 VCP",  width="small"),
                "Trap":    st.column_config.TextColumn("🪤 Trap", width="medium"),
                "RS":      st.column_config.NumberColumn("💪 RS", format="%.2f"),
                "RS_Lead": st.column_config.TextColumn("Lead",    width="small"),
                "Patterns":st.column_config.TextColumn("Patterns",width="large"),
            })

# ── Metrics ──────────────────────────────────────────────────────────────────
elif _page == 'metrics':
    a = calc_analytics(df)
    if not a or a.get("closed_trades", 0) == 0:
        st.info("Metrics require historical closed trades.")
    else:
        st.markdown('<div class="sec">Strategy Performance Metrics</div>',
                    unsafe_allow_html=True)
        st.markdown(
            '<div class="cards">'
            + card("Win Rate",      f'{a["win_rate"]}%',
                   f'{a["wins"]}W / {a["losses"]}L',
                   "green" if a["win_rate"] >= 50 else "red")
            + card("Profit Factor", str(a["profit_factor"]), "Gross P / Gross L")
            + card("Expectancy",    f'${a["expectancy"]}')
            + card("Avg Win",       f'${a["avg_win"]:,.0f}')
            + card("Avg Loss",      f'${a["avg_loss"]:,.0f}', "", "red")
            + card("Max Drawdown",  f'${a["max_drawdown"]:,.0f}', "", "red")
            + card("Avg Hold",      f'{a["avg_hold_days"]}d')
            + card("Sharpe",        str(a["sharpe"]))
            + '</div>',
            unsafe_allow_html=True)

# ── Watchlist ────────────────────────────────────────────────────────────────
elif _page == 'watchlist':
    st.markdown('<div class="sec">👁 Target Watchlist</div>', unsafe_allow_html=True)

    def drop_watchlist_cb(w_id, s_name):
        delete_watchlist_item(w_id, UID)
        st.toast(f"🗑️ Dropped {s_name}")

    with st.form(key="add_stock_form", clear_on_submit=True):
        col_inp, col_btn = st.columns([4, 1])
        with col_inp:
            new_stock = st.text_input(
                "Stock Ticker", placeholder="e.g., SBIN, TATAMOTORS",
                label_visibility="collapsed").upper().strip()
        with col_btn:
            if st.form_submit_button("➕ Add", width="stretch") and new_stock:
                add_watchlist(UID, new_stock)
                st.toast(f"🚀 {new_stock} added!")
                st.rerun()

    wdf = get_watchlist(UID)
    if not wdf.empty:
        st.markdown('<div class="sec" style="margin-top:1rem">Live Monitored Assets</div>',
                    unsafe_allow_html=True)
        wl_symbols = wdf["stock"].tolist()
        with st.spinner("Fetching live metrics..."):
            wl_data = _bulk_fetch_history(wl_symbols, period="3mo")

        cols = st.columns(3)
        for i, row in wdf.iterrows():
            stock = row["stock"]
            wid   = int(row["id"])
            col   = cols[i % 3]
            with col:
                df_hist = wl_data.get(stock)
                ind = compute_indicators(stock, period="3mo", prefetched_df=df_hist)
                if ind:
                    cmp_v  = ind.get("cmp", "—")
                    rsi_v  = ind.get("rsi", "—")
                    trend  = ind.get("trend", "—")
                    sup    = ind.get("support", "—")
                    res    = ind.get("resistance", "—")
                    ema9   = ind.get("ema9",  ind.get("ema20", "—"))
                    ema21  = ind.get("ema21", ind.get("ema50", "—"))
                    brd = (theme_t["green"]  if "Uptrend"   in str(trend)
                           else theme_t["red"] if "Downtrend" in str(trend)
                           else theme_t["yellow"])
                    st.markdown(f"""
<div style="background:var(--card);border-top:4px solid {brd};border-radius:8px;
     padding:1rem;box-shadow:0 4px 6px rgba(0,0,0,.05);margin-bottom:.5rem">
  <div style="font-size:1.1rem;font-weight:800;color:var(--text)">{stock}</div>
  <div style="font-size:.75rem;color:var(--muted);margin-bottom:.5rem;
       text-transform:uppercase">{get_sector(stock)}</div>
  <div style="font-size:.8rem;line-height:1.6;color:var(--text)">
    <b>CMP:</b> ${cmp_v}<br>
    <b>RSI:</b> {rsi_v} | <b>Trend:</b> {trend}<br>
    <b>EMA9:</b> ${ema9} | <b>EMA21:</b> ${ema21}<br>
    <b>Sup:</b> ${sup} | <b>Res:</b> ${res}
  </div>
</div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
<div style="background:var(--card);border-top:4px solid var(--muted);
     border-radius:8px;padding:1rem;margin-bottom:.5rem">
  <div style="font-size:1.1rem;font-weight:800;color:var(--text)">{stock}</div>
  <div style="font-size:.85rem;color:var(--red);margin-bottom:.5rem">
    Data Unavailable — check symbol</div>
</div>""", unsafe_allow_html=True)

                st.button("🗑️ Drop", key=f"wl_del_{wid}",
                          on_click=drop_watchlist_cb,
                          args=(wid, stock), width="stretch")
    else:
        st.info("Watchlist empty. Add a ticker above.")

# ── Export ───────────────────────────────────────────────────────────────────
elif _page == 'export':
    if df.empty:
        st.info("No data available for export.")
    else:
        st.markdown('<div class="sec">Raw Database Export</div>',
                    unsafe_allow_html=True)
        st.dataframe(df)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download CSV", csv,
            file_name=f"swing_portfolio_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv")

# ── Signal Scores ────────────────────────────────────────────────────────────
elif _page == 'scores':
    st.markdown('<div class="sec">🎯 signals.py — Component Scorecard</div>',
                unsafe_allow_html=True)
    render_score_dashboard()
    st.markdown("""
<div style="margin-top:1rem;padding:1rem;background:rgba(16,185,129,.08);
     border:1px solid rgba(16,185,129,.3);border-radius:8px;font-size:.85rem;
     color:var(--muted);line-height:1.8">
<b style="color:var(--text)">✅ Engine capabilities:</b><br>
1. <b>Core indicators</b> — Wilder RSI/ATR, single-pass MACD, numpy Supertrend, clamped Bollinger, 20-day VWAP, swing-peak Fibonacci<br>
2. <b>Risk engine</b> — unified <code>_calc_risk_params</code> across signals, picks, and scanner (zero phantom RR)<br>
3. <b>Trap scanner</b> — 5-factor bull/bear trap confluence across the full universe<br>
4. <b>Smart Money (SMC)</b> — FVG, order blocks, liquidity pools, premium/discount, displacement<br>
5. <b>VCP</b> — Minervini volatility-contraction base detection with pivot-ready flagging<br>
6. <b>Relative Strength</b> — IBD-style RS ratio + 1-99 percentile leadership rating vs S&P<br>
7. <b>Unified in scanner</b> — VCP, Trap, and RS now surface as columns + filters in the Universe Scanner
</div>""", unsafe_allow_html=True)

# ── Trap Scanner ─────────────────────────────────────────────────────────────
elif _page == 'traps':
    if not _TRAP_SCANNER_AVAILABLE:
        st.warning("🪤 Trap Scanner requires the updated **signals.py** (v12+). "
                   "Deploy the new signals.py from the project outputs to enable this tab.",
                   icon="⚠️")
    st.markdown('<div class="sec">🪤 Bull & Bear Trap Scanner — Full US universe</div>',
                unsafe_allow_html=True)

    # ── Summary banner ─────────────────────────────────────────────────────────
    trap_data = st.session_state.trap_scan_cache
    if trap_data:
        bull_n = trap_data.get("bull_count", 0)
        bear_n = trap_data.get("bear_count", 0)
        scanned = trap_data.get("scanned", 0)
        liquid  = trap_data.get("liquid", 0)
        ts      = trap_data.get("timestamp", "—")
        st.markdown(
            f'<div style="display:flex;gap:1rem;flex-wrap:wrap;margin-bottom:1.5rem">'
            f'<div style="background:rgba(239,68,68,.1);border:1px solid rgba(239,68,68,.3);'
            f'border-radius:10px;padding:.7rem 1.2rem;font-weight:800;font-size:.9rem">'
            f'🔴 Bull Traps: <span style="color:var(--red)">{bull_n}</span></div>'
            f'<div style="background:rgba(16,185,129,.1);border:1px solid rgba(16,185,129,.3);'
            f'border-radius:10px;padding:.7rem 1.2rem;font-weight:800;font-size:.9rem">'
            f'🟢 Bear Traps: <span style="color:var(--green)">{bear_n}</span></div>'
            f'<div style="background:var(--card);border:1px solid var(--border);'
            f'border-radius:10px;padding:.7rem 1.2rem;font-size:.8rem;color:var(--muted);font-weight:600">'
            f'🔍 Scanned: {scanned} | Liquid: {liquid} | Updated: {ts}</div>'
            f'</div>',
            unsafe_allow_html=True)

    # ── Controls ────────────────────────────────────────────────────────────────
    ctrl1, ctrl2, ctrl3 = st.columns([2, 1, 1])
    with ctrl1:
        st.caption("⚡ Sweeps all US universe liquid stocks for false breakout / breakdown patterns.")
    with ctrl2:
        min_conf = st.slider("Min Confidence %", 50, 90, 60, 5, label_visibility="collapsed")
    with ctrl3:
        run_trap_scan = st.button("🪤 Run Trap Scan", width="stretch")

    if run_trap_scan:
        total_sym = min(MAX_SCAN_SYMBOLS, len(SECTOR_MAP))
        with st.spinner(f"🔍 Scanning {total_sym} stocks for trap patterns…"):
            st.session_state.trap_scan_cache = scan_for_traps(min_confidence=min_conf)
            trap_data = st.session_state.trap_scan_cache
            st.toast(
                f"✅ Found {trap_data['bull_count']} bull traps, "
                f"{trap_data['bear_count']} bear traps across {trap_data['liquid']} liquid stocks",
                icon="🪤")

    if not trap_data:
        st.info("💡 Click **🪤 Run Trap Scan** to sweep the full US universe for active trap patterns.")
    else:
        bull_traps = trap_data.get("bull_traps", [])
        bear_traps = trap_data.get("bear_traps", [])

        # ── Filter by confidence slider ─────────────────────────────────────────
        bull_traps = [x for x in bull_traps if x["confidence"] >= min_conf]
        bear_traps = [x for x in bear_traps if x["confidence"] >= min_conf]

        col_bull, col_bear = st.columns(2)

        # ── BULL TRAPS ──────────────────────────────────────────────────────────
        with col_bull:
            st.markdown(
                f'<div style="font-size:.85rem;font-weight:800;color:var(--red);'
                f'text-transform:uppercase;letter-spacing:.1em;margin-bottom:.8rem;'
                f'padding:.5rem .8rem;background:rgba(239,68,68,.08);'
                f'border-left:4px solid var(--red);border-radius:0 8px 8px 0">'
                f'🔴 Bull Traps — Exit / Avoid ({len(bull_traps)})</div>',
                unsafe_allow_html=True)

            if not bull_traps:
                st.success("✅ No bull traps found at this confidence level.")
            else:
                for bt in bull_traps:
                  try:
                    conf = bt["confidence"]
                    conf_clr = "#ef4444" if conf >= 80 else "#f59e0b"
                    st.markdown(f"""
<div style="background:var(--card);border:1px solid rgba(239,68,68,.25);
     border-left:4px solid var(--red);border-radius:10px;
     padding:1rem 1.2rem;margin-bottom:.8rem;
     box-shadow:0 4px 12px -4px rgba(239,68,68,.15)">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:.4rem">
    <span style="font-weight:800;font-size:.95rem">{bt['stock']}</span>
    <span style="background:rgba(239,68,68,.12);color:{conf_clr};
          padding:.2rem .6rem;border-radius:6px;font-size:.75rem;font-weight:800">
      {conf}% CONF
    </span>
  </div>
  <div style="font-size:.75rem;color:var(--muted);margin-bottom:.5rem">
    {bt['sector']} · CMP ${bt['cmp']} · RSI {bt['rsi'] if bt['rsi'] else '—'}
  </div>
  <div style="font-size:.8rem;color:var(--red);font-weight:600;margin-bottom:.5rem">
    ⚠️ {bt['detail']}
  </div>
  <div style="height:4px;background:var(--input);border-radius:2px;margin-bottom:.6rem">
    <div style="height:4px;border-radius:2px;background:var(--red);width:{min(conf,100)}%"></div>
  </div>
  <div style="font-size:.78rem;color:var(--muted);display:grid;grid-template-columns:1fr 1fr;gap:.2rem">
    <span>📊 Trend: {bt['trend']}</span>
    <span>📦 Vol: {bt['vol_ratio']:.1f}x avg</span>
    <span>🛡 Support: ${bt['support']}</span>
    <span>🚧 Resist: ${bt['resistance']}</span>
    <span>🔁 Re-entry SL: ${bt['re_entry_sl']}</span>
    <span>ST: {'🟢 Bull' if bt.get('supertrend_bullish') else '🔴 Bear'}</span>
  </div>
  {('<div style="font-size:.72rem;color:var(--muted);margin-top:.4rem">📐 ' + bt['patterns'] + '</div>') if bt.get('patterns') else ''}
</div>""", unsafe_allow_html=True)
                  except Exception:
                    st.markdown(
                        f'<div style="background:var(--card);border:1px solid var(--border);'
                        f'border-radius:8px;padding:.7rem 1rem;margin-bottom:.6rem;font-size:.85rem">'
                        f'<b>{bt.get("stock","?")}</b> — bull trap '
                        f'{bt.get("confidence","?")}% (detail unavailable)</div>',
                        unsafe_allow_html=True)

        # ── BEAR TRAPS ──────────────────────────────────────────────────────────
        with col_bear:
            st.markdown(
                f'<div style="font-size:.85rem;font-weight:800;color:var(--green);'
                f'text-transform:uppercase;letter-spacing:.1em;margin-bottom:.8rem;'
                f'padding:.5rem .8rem;background:rgba(16,185,129,.08);'
                f'border-left:4px solid var(--green);border-radius:0 8px 8px 0">'
                f'🟢 Bear Traps — Buy Opportunity ({len(bear_traps)})</div>',
                unsafe_allow_html=True)

            if not bear_traps:
                st.info("No bear traps found at this confidence level.")
            else:
                for brt in bear_traps:
                  try:
                    conf = brt["confidence"]
                    conf_clr = "#10b981" if conf >= 80 else "#f59e0b"
                    rr = brt.get("risk_reward")
                    rr_str = f"R:R {rr}" if rr else "—"
                    st.markdown(f"""
<div style="background:var(--card);border:1px solid rgba(16,185,129,.25);
     border-left:4px solid var(--green);border-radius:10px;
     padding:1rem 1.2rem;margin-bottom:.8rem;
     box-shadow:0 4px 12px -4px rgba(16,185,129,.15)">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:.4rem">
    <span style="font-weight:800;font-size:.95rem">{brt['stock']}</span>
    <span style="background:rgba(16,185,129,.12);color:{conf_clr};
          padding:.2rem .6rem;border-radius:6px;font-size:.75rem;font-weight:800">
      {conf}% CONF
    </span>
  </div>
  <div style="font-size:.75rem;color:var(--muted);margin-bottom:.5rem">
    {brt['sector']} · CMP ${brt['cmp']} · RSI {brt['rsi'] if brt['rsi'] else '—'}
  </div>
  <div style="font-size:.8rem;color:var(--green);font-weight:600;margin-bottom:.5rem">
    🪤 {brt['detail']}
  </div>
  <div style="height:4px;background:var(--input);border-radius:2px;margin-bottom:.6rem">
    <div style="height:4px;border-radius:2px;background:var(--green);width:{min(conf,100)}%"></div>
  </div>
  <div style="background:rgba(16,185,129,.06);border-radius:6px;
       padding:.6rem .8rem;margin-bottom:.5rem;
       display:grid;grid-template-columns:1fr 1fr 1fr;gap:.3rem;font-size:.8rem;font-weight:700">
    <span>🎯 Entry<br><b>${brt['entry']}</b></span>
    <span>🚀 Target<br><b style="color:var(--green)">${brt['target']}</b></span>
    <span>🛑 SL<br><b style="color:var(--red)">${brt['stop_loss']}</b></span>
  </div>
  <div style="font-size:.78rem;color:var(--muted);display:grid;grid-template-columns:1fr 1fr;gap:.2rem">
    <span>📊 {rr_str}</span>
    <span>📦 Vol: {brt['vol_ratio']:.1f}x avg</span>
    <span>🛡 Support: ${brt['support']}</span>
    <span>🚧 Resist: ${brt['resistance']}</span>
    <span>📈 Trend: {brt['trend']}</span>
    <span>ST: {'🟢 Bull' if brt.get('supertrend_bullish') else '🔴 Bear'}</span>
  </div>
  {('<div style="font-size:.72rem;color:var(--muted);margin-top:.4rem">📐 ' + brt['patterns'] + '</div>') if brt.get('patterns') else ''}
</div>""", unsafe_allow_html=True)
                  except Exception:
                    st.markdown(
                        f'<div style="background:var(--card);border:1px solid var(--border);'
                        f'border-radius:8px;padding:.7rem 1rem;margin-bottom:.6rem;font-size:.85rem">'
                        f'<b>{brt.get("stock","?")}</b> — bear trap '
                        f'{brt.get("confidence","?")}% (detail unavailable)</div>',
                        unsafe_allow_html=True)

        # ── Export trap results ─────────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        if bull_traps or bear_traps:
            bull_df = pd.DataFrame(bull_traps)[
                ["stock","sector","cmp","rsi","confidence","detail","support","resistance","trend"]
            ] if bull_traps else pd.DataFrame()
            bear_df = pd.DataFrame(bear_traps)[
                ["stock","sector","cmp","rsi","confidence","detail","entry","target","stop_loss","risk_reward","trend"]
            ] if bear_traps else pd.DataFrame()

            exp1, exp2 = st.columns(2)
            with exp1:
                if not bull_df.empty:
                    st.download_button(
                        "⬇️ Export Bull Traps CSV",
                        bull_df.to_csv(index=False).encode("utf-8"),
                        file_name=f"bull_traps_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv", use_container_width=True)
            with exp2:
                if not bear_df.empty:
                    st.download_button(
                        "⬇️ Export Bear Traps CSV",
                        bear_df.to_csv(index=False).encode("utf-8"),
                        file_name=f"bear_traps_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv", use_container_width=True)

# ── Corporate Actions ────────────────────────────────────────────────────────
elif _page == 'corp_actions':
    if not _CORP_ACTIONS_AVAILABLE:
        st.warning("📅 Corporate Actions requires the updated **signals.py** (v12+). "
                   "Deploy the new signals.py from the project outputs to enable this tab.",
                   icon="⚠️")
    st.markdown('<div class="sec">📅 Corporate Actions — Full US universe</div>',
                unsafe_allow_html=True)
    st.caption("Dividends · Stock Splits · Bonus Issues — sourced from NSE via yfinance. 6-hour cache.")

    # ── Portfolio holdings quick-view ──────────────────────────────────────────
    open_syms = raw[raw["status"]=="Open"]["stock"].unique().tolist() if not raw.empty else []
    if open_syms:
        st.markdown('<div class="sec" style="margin-top:1rem">📌 Your Holdings</div>',
                    unsafe_allow_html=True)
        with st.spinner("Fetching corporate actions for your holdings..."):
            port_actions = fetch_bulk_corporate_actions(open_syms, max_workers=5)

        p_rows = ""
        for sym in open_syms:
            data = port_actions.get(sym, {})
            div_str  = (f"${data['last_dividend']} on {data['last_div_date']}"
                        if data.get("last_dividend") else "—")
            exd_str  = (f'<b style="color:var(--yellow)">{data["upcoming_exdate"]}</b>'
                        if data.get("upcoming_exdate") else "—")
            spl_str  = (f"{data['splits'][-1]['ratio']}x on {data['splits'][-1]['date']}"
                        if data.get("splits") else "—")
            split_badge = ('<span style="background:rgba(59,130,246,.15);color:var(--blue);'
                           'padding:.1rem .5rem;border-radius:4px;font-size:.7rem;'
                           'font-weight:800">SPLIT/BONUS 1Y</span>'
                           if data.get("has_split_1y") else "")
            p_rows += (
                f"<tr>"
                f"<td style='text-align:left;font-weight:800'>{sym} {split_badge}</td>"
                f"<td style='text-align:left;color:var(--muted);font-size:.8rem'>"
                f"{get_sector(sym)}</td>"
                f"<td>{div_str}</td>"
                f"<td>{exd_str}</td>"
                f"<td style='font-size:.78rem;color:var(--muted)'>{spl_str}</td>"
                f"</tr>"
            )
        if p_rows:
            st.markdown(
                f'<div class="tbl-wrap"><table class="t">'
                f'<thead><tr>'
                f'<th class="l">Stock</th><th class="l">Sector</th>'
                f'<th>Last Dividend</th><th>Ex-Date (upcoming)</th>'
                f'<th>Last Split/Bonus</th>'
                f'</tr></thead><tbody>{p_rows}</tbody></table></div>',
                unsafe_allow_html=True)

    st.markdown("<hr style='border-color:var(--border);margin:1.5rem 0'>",
                unsafe_allow_html=True)

    # ── Full universe scan controls ────────────────────────────────────────────
    ca1, ca2 = st.columns([3, 1])
    with ca1:
        st.markdown(
            '<div style="font-size:.85rem;font-weight:700;color:var(--text)">'
            '🔍 Sweep full US universe for upcoming ex-dates, recent dividends '
            'and bonus/split events</div>',
            unsafe_allow_html=True)
    with ca2:
        run_ca_scan = st.button("📅 Scan Corporate Actions", width="stretch")

    if run_ca_scan:
        total_sym = min(MAX_SCAN_SYMBOLS, len(SECTOR_MAP))
        with st.spinner(f"Fetching corporate actions for {total_sym} stocks… (may take 60–90s)"):
            st.session_state.corp_actions_cache = scan_corporate_actions_universe()
            ca = st.session_state.corp_actions_cache
            st.toast(
                f"✅ {len(ca['with_upcoming_exdate'])} upcoming ex-dates · "
                f"{len(ca['recent_dividends'])} recent dividends · "
                f"{len(ca['recent_splits'])} splits/bonus",
                icon="📅")

    ca_data = st.session_state.corp_actions_cache
    if ca_data is None:
        st.info("Click **📅 Scan Corporate Actions** to fetch the full US universe action calendar.")
    else:
        ts = ca_data.get("timestamp","—"); scanned = ca_data.get("scanned",0)
        st.markdown(
            f'<div style="font-size:.75rem;color:var(--muted);margin-bottom:1rem">'
            f'Last scanned {scanned} stocks at {ts}</div>',
            unsafe_allow_html=True)

        ca_t1, ca_t2, ca_t3 = st.tabs([
            f"📆 Upcoming Ex-Dates ({len(ca_data['with_upcoming_exdate'])})",
            f"💰 Recent Dividends ({len(ca_data['recent_dividends'])})",
            f"🔀 Splits & Bonus ({len(ca_data['recent_splits'])})",
        ])

        # ── Upcoming Ex-Dates ──────────────────────────────────────────────────
        with ca_t1:
            exd_list = ca_data["with_upcoming_exdate"]
            if not exd_list:
                st.info("No upcoming ex-dividend dates found.")
            else:
                rows = ""
                for item in exd_list:
                    days_away = (pd.Timestamp(item["ex_date"]) -
                                 pd.Timestamp.now()).days
                    urgency = (
                        f'<span style="color:var(--red);font-weight:800">'
                        f'⚡ {days_away}d away</span>'
                        if days_away <= 7 else
                        f'<span style="color:var(--yellow);font-weight:700">'
                        f'{days_away}d</span>'
                        if days_away <= 30 else
                        f'<span style="color:var(--muted)">{days_away}d</span>'
                    )
                    div_amt = (f"${item['last_dividend']}"
                               if item.get("last_dividend") else "—")
                    rows += (
                        f"<tr>"
                        f"<td style='text-align:left;font-weight:800'>{item['stock']}</td>"
                        f"<td style='text-align:left;color:var(--muted);font-size:.8rem'>"
                        f"{item['sector']}</td>"
                        f"<td><b>{item['ex_date']}</b></td>"
                        f"<td>{urgency}</td>"
                        f"<td>{div_amt}</td>"
                        f"</tr>"
                    )
                st.markdown(
                    f'<div class="tbl-wrap"><table class="t"><thead><tr>'
                    f'<th class="l">Stock</th><th class="l">Sector</th>'
                    f'<th>Ex-Date</th><th>Days Away</th><th>Last Div Amt</th>'
                    f'</tr></thead><tbody>{rows}</tbody></table></div>',
                    unsafe_allow_html=True)
                # Export
                ex_df = pd.DataFrame(exd_list)
                st.download_button(
                    "⬇️ Export Ex-Dates CSV",
                    ex_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"upcoming_exdates_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv")

        # ── Recent Dividends ───────────────────────────────────────────────────
        with ca_t2:
            div_list = ca_data["recent_dividends"]
            if not div_list:
                st.info("No recent dividends found in the last 12 months.")
            else:
                rows = ""
                for item in sorted(div_list, key=lambda x: x["amount"], reverse=True):
                    rows += (
                        f"<tr>"
                        f"<td style='text-align:left;font-weight:800'>{item['stock']}</td>"
                        f"<td style='text-align:left;color:var(--muted);font-size:.8rem'>"
                        f"{item['sector']}</td>"
                        f"<td><b style='color:var(--green)'>${item['amount']}</b></td>"
                        f"<td>{item['ex_date']}</td>"
                        f"</tr>"
                    )
                st.markdown(
                    f'<div class="tbl-wrap"><table class="t"><thead><tr>'
                    f'<th class="l">Stock</th><th class="l">Sector</th>'
                    f'<th>Dividend $</th><th>Ex-Date</th>'
                    f'</tr></thead><tbody>{rows}</tbody></table></div>',
                    unsafe_allow_html=True)
                div_df = pd.DataFrame(div_list)
                st.download_button(
                    "⬇️ Export Dividends CSV",
                    div_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"recent_dividends_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv")

        # ── Splits & Bonus ─────────────────────────────────────────────────────
        with ca_t3:
            split_list = ca_data["recent_splits"]
            if not split_list:
                st.info("No stock splits or bonus issues found in the last 12 months.")
            else:
                rows = ""
                for item in split_list:
                    type_badge = (
                        f'<span style="background:rgba(59,130,246,.15);color:var(--blue);'
                        f'padding:.2rem .6rem;border-radius:5px;font-size:.72rem;'
                        f'font-weight:800">{item["type"]}</span>'
                    )
                    rows += (
                        f"<tr>"
                        f"<td style='text-align:left;font-weight:800'>{item['stock']}</td>"
                        f"<td style='text-align:left;color:var(--muted);font-size:.8rem'>"
                        f"{item['sector']}</td>"
                        f"<td>{type_badge}</td>"
                        f"<td><b>{item['ratio']}:1</b></td>"
                        f"<td>{item['date']}</td>"
                        f"</tr>"
                    )
                st.markdown(
                    f'<div class="tbl-wrap"><table class="t"><thead><tr>'
                    f'<th class="l">Stock</th><th class="l">Sector</th>'
                    f'<th>Type</th><th>Ratio</th><th>Date</th>'
                    f'</tr></thead><tbody>{rows}</tbody></table></div>',
                    unsafe_allow_html=True)
                sp_df = pd.DataFrame(split_list)
                st.download_button(
                    "⬇️ Export Splits/Bonus CSV",
                    sp_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"splits_bonus_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv")

# ── Smart Money Concepts (SMC / ICT) ──────────────────────────────────────────
elif _page == 'smc':
    st.markdown('<div class="sec">🏦 Smart Money Concepts — FVG · Order Blocks · Liquidity</div>',
                unsafe_allow_html=True)
    st.caption("Institutional footprint analysis: Fair Value Gaps, Order Blocks, "
               "Liquidity Pools, Premium/Discount zones, and Displacement. "
               "Optimised for NSE daily charts with circuit-filter awareness.")

    # ── Stock selector: portfolio holdings + custom symbol ─────────────────────
    open_syms = (raw[raw["status"]=="Open"]["stock"].unique().tolist()
                 if not raw.empty else [])
    sc1, sc2 = st.columns([2, 1])
    with sc1:
        symbol_options = open_syms + ["— Enter custom symbol —"]
        sel_sym = st.selectbox("Select stock for SMC analysis", symbol_options,
                               label_visibility="collapsed")
    with sc2:
        custom_sym = st.text_input("Custom", placeholder="e.g. RELIANCE",
                                   label_visibility="collapsed")

    target_sym = (custom_sym.strip().upper() if custom_sym.strip()
                  else (sel_sym if sel_sym != "— Enter custom symbol —" else None))

    if not target_sym:
        st.info("💡 Select a holding or enter any NSE symbol to see its Smart Money structure.")
    else:
        with st.spinner(f"Analysing {target_sym} institutional structure…"):
            try:
                ind = compute_indicators(target_sym, period="6mo")
            except Exception as e:
                ind = None
                st.error(f"Could not analyse {target_sym}: {e}")

        if ind:
            cmp = ind.get("cmp", 0)
            score = ind.get("smc_score", 0)
            label = ind.get("smc_label", "Neutral SMC")
            zone  = ind.get("smc_zone", "Unknown")
            bias  = ind.get("smc_bias", "Neutral")
            action = ind.get("smc_action", "WAIT")
            entry  = ind.get("smc_entry")
            target = ind.get("smc_target")
            sl     = ind.get("smc_sl")
            rr     = ind.get("smc_rr")
            quality = ind.get("smc_setup_quality")
            reason = ind.get("smc_setup_reason", "")

            # ── ACTIONABLE TRADE SETUP CARD (the headline) ─────────────────────
            if action in ("BUY", "SELL") and entry:
                act_clr = theme_t["green"] if action == "BUY" else theme_t["red"]
                act_bg  = ("rgba(16,185,129,.08)" if action == "BUY"
                           else "rgba(239,68,68,.08)")
                q_clr = ("#fbbf24" if quality == "A+" else
                         theme_t["accent"] if quality == "A" else theme_t["muted"])
                st.markdown(
                    f'<div style="background:{act_bg};border:2px solid {act_clr};'
                    f'border-radius:14px;padding:1.4rem;margin:1rem 0">'
                    f'<div style="display:flex;justify-content:space-between;'
                    f'align-items:center;margin-bottom:1rem">'
                    f'<div style="font-size:1.8rem;font-weight:800;color:{act_clr}">'
                    f'{"🟢" if action=="BUY" else "🔴"} {action} {target_sym}</div>'
                    f'<div style="background:{q_clr};color:#000;padding:.3rem .9rem;'
                    f'border-radius:8px;font-size:.95rem;font-weight:800">'
                    f'{quality} Setup</div></div>'
                    f'<div style="display:grid;grid-template-columns:repeat(4,1fr);'
                    f'gap:.8rem;margin-bottom:1rem">'
                    f'<div style="background:var(--card);border-radius:10px;padding:.9rem;'
                    f'text-align:center"><div style="font-size:.7rem;color:var(--muted);'
                    f'font-weight:700;text-transform:uppercase">Entry</div>'
                    f'<div style="font-size:1.3rem;font-weight:800;color:var(--text)">'
                    f'${entry}</div></div>'
                    f'<div style="background:var(--card);border-radius:10px;padding:.9rem;'
                    f'text-align:center"><div style="font-size:.7rem;color:var(--muted);'
                    f'font-weight:700;text-transform:uppercase">Target</div>'
                    f'<div style="font-size:1.3rem;font-weight:800;color:{theme_t["green"]}">'
                    f'${target}</div></div>'
                    f'<div style="background:var(--card);border-radius:10px;padding:.9rem;'
                    f'text-align:center"><div style="font-size:.7rem;color:var(--muted);'
                    f'font-weight:700;text-transform:uppercase">Stop Loss</div>'
                    f'<div style="font-size:1.3rem;font-weight:800;color:{theme_t["red"]}">'
                    f'${sl}</div></div>'
                    f'<div style="background:var(--card);border-radius:10px;padding:.9rem;'
                    f'text-align:center"><div style="font-size:.7rem;color:var(--muted);'
                    f'font-weight:700;text-transform:uppercase">Risk:Reward</div>'
                    f'<div style="font-size:1.3rem;font-weight:800;color:var(--accent)">'
                    f'1:{rr}</div></div>'
                    f'</div>'
                    f'<div style="font-size:.82rem;color:var(--muted);line-height:1.6">'
                    f'📋 {reason}</div>'
                    f'</div>',
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<div style="background:var(--card);border:2px solid var(--border);'
                    f'border-radius:14px;padding:1.4rem;margin:1rem 0;text-align:center">'
                    f'<div style="font-size:1.5rem;font-weight:800;color:var(--muted)">'
                    f'⏸ WAIT — {target_sym}</div>'
                    f'<div style="font-size:.85rem;color:var(--muted);margin-top:.5rem">'
                    f'{reason}</div></div>',
                    unsafe_allow_html=True)

            # ── Context cards (now secondary, below the action) ────────────────
            score_clr = (theme_t["green"] if score >= 35 else
                         theme_t["red"] if score <= -35 else theme_t["muted"])
            zone_clr  = (theme_t["red"] if zone == "Premium" else
                         theme_t["green"] if zone == "Discount" else theme_t["muted"])
            st.markdown(
                f'<div style="display:flex;gap:1rem;flex-wrap:wrap;margin:1rem 0">'
                f'<div style="flex:1;min-width:180px;background:var(--card);'
                f'border:1px solid {score_clr};border-radius:12px;padding:1.2rem">'
                f'<div style="font-size:.7rem;color:var(--muted);font-weight:700;'
                f'text-transform:uppercase;letter-spacing:.08em">SMC Bias</div>'
                f'<div style="font-size:1.5rem;font-weight:800;color:{score_clr};'
                f'margin:.2rem 0">{label}</div>'
                f'<div style="font-size:.8rem;color:var(--muted)">Score: {score:+d} / 100</div>'
                f'</div>'
                f'<div style="flex:1;min-width:180px;background:var(--card);'
                f'border:1px solid {zone_clr};border-radius:12px;padding:1.2rem">'
                f'<div style="font-size:.7rem;color:var(--muted);font-weight:700;'
                f'text-transform:uppercase;letter-spacing:.08em">Premium/Discount</div>'
                f'<div style="font-size:1.5rem;font-weight:800;color:{zone_clr};'
                f'margin:.2rem 0">{zone}</div>'
                f'<div style="font-size:.8rem;color:var(--muted)">'
                f'{ind.get("smc_zone_pct","—")}% of range · {bias} bias</div>'
                f'</div>'
                f'<div style="flex:1;min-width:180px;background:var(--card);'
                f'border:1px solid var(--border);border-radius:12px;padding:1.2rem">'
                f'<div style="font-size:.7rem;color:var(--muted);font-weight:700;'
                f'text-transform:uppercase;letter-spacing:.08em">CMP</div>'
                f'<div style="font-size:1.5rem;font-weight:800;color:var(--text);'
                f'margin:.2rem 0">${cmp}</div>'
                f'<div style="font-size:.8rem;color:var(--muted)">'
                f'Range ${ind.get("smc_range_low","—")}–${ind.get("smc_range_high","—")}</div>'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True)

            # ── Displacement banner ────────────────────────────────────────────
            disp = ind.get("smc_displacement")
            if disp:
                d_ago = ind.get("smc_displacement_bars_ago", "?")
                d_clr = theme_t["green"] if disp == "Bullish" else theme_t["red"]
                st.markdown(
                    f'<div style="background:rgba(0,0,0,.15);border-left:4px solid {d_clr};'
                    f'border-radius:0 8px 8px 0;padding:.7rem 1rem;margin-bottom:1rem;'
                    f'font-size:.85rem">⚡ <b style="color:{d_clr}">{disp} Displacement</b> '
                    f'detected {d_ago} bar(s) ago — institutional momentum present.</div>',
                    unsafe_allow_html=True)

            # ── Four detail panels ─────────────────────────────────────────────
            colA, colB = st.columns(2)

            # FVG panel
            with colA:
                st.markdown('<div style="font-size:.85rem;font-weight:800;color:var(--text);'
                            'margin-bottom:.5rem">📊 Fair Value Gaps</div>',
                            unsafe_allow_html=True)
                nbf = ind.get("smc_nearest_bull_fvg")
                nbef = ind.get("smc_nearest_bear_fvg")
                in_bull = ind.get("smc_in_bull_fvg")
                in_bear = ind.get("smc_in_bear_fvg")
                fvg_rows = ""
                if in_bull:
                    fvg_rows += ('<div style="color:var(--green);font-size:.82rem;'
                                 'margin-bottom:.3rem">📍 Price currently INSIDE a bullish FVG (support)</div>')
                if in_bear:
                    fvg_rows += ('<div style="color:var(--red);font-size:.82rem;'
                                 'margin-bottom:.3rem">📍 Price currently INSIDE a bearish FVG (resistance)</div>')
                if nbf:
                    fvg_rows += (f'<div style="font-size:.82rem;margin-bottom:.3rem">'
                                 f'🟢 Nearest bull FVG below: <b>${nbf["bottom"]}–${nbf["top"]}</b> '
                                 f'({nbf["size_atr"]} ATR)</div>')
                if nbef:
                    fvg_rows += (f'<div style="font-size:.82rem;margin-bottom:.3rem">'
                                 f'🔴 Nearest bear FVG above: <b>${nbef["bottom"]}–${nbef["top"]}</b> '
                                 f'({nbef["size_atr"]} ATR)</div>')
                fvg_rows += (f'<div style="font-size:.75rem;color:var(--muted);margin-top:.4rem">'
                             f'Unfilled: {ind.get("smc_bull_fvg_count",0)} bullish · '
                             f'{ind.get("smc_bear_fvg_count",0)} bearish</div>')
                if not (nbf or nbef or in_bull or in_bear):
                    fvg_rows = '<div style="font-size:.82rem;color:var(--muted)">No significant unfilled FVGs nearby.</div>'
                st.markdown(f'<div style="background:var(--card);border:1px solid var(--border);'
                            f'border-radius:10px;padding:1rem">{fvg_rows}</div>',
                            unsafe_allow_html=True)

            # Order Block panel
            with colB:
                st.markdown('<div style="font-size:.85rem;font-weight:800;color:var(--text);'
                            'margin-bottom:.5rem">🧱 Order Blocks</div>',
                            unsafe_allow_html=True)
                nbo = ind.get("smc_nearest_bull_ob")
                nbeo = ind.get("smc_nearest_bear_ob")
                ob_rows = ""
                if ind.get("smc_at_bull_ob"):
                    ob_rows += ('<div style="color:var(--green);font-size:.82rem;'
                                'margin-bottom:.3rem">📍 Price at a bullish order block (demand)</div>')
                if ind.get("smc_at_bear_ob"):
                    ob_rows += ('<div style="color:var(--red);font-size:.82rem;'
                                'margin-bottom:.3rem">📍 Price at a bearish order block (supply)</div>')
                if nbo:
                    ob_rows += (f'<div style="font-size:.82rem;margin-bottom:.3rem">'
                                f'🟢 Bull OB (demand): <b>${nbo["bottom"]}–${nbo["top"]}</b> '
                                f'({nbo["strength_atr"]} ATR move)</div>')
                if nbeo:
                    ob_rows += (f'<div style="font-size:.82rem;margin-bottom:.3rem">'
                                f'🔴 Bear OB (supply): <b>${nbeo["bottom"]}–${nbeo["top"]}</b> '
                                f'({nbeo["strength_atr"]} ATR move)</div>')
                if not (nbo or nbeo or ind.get("smc_at_bull_ob") or ind.get("smc_at_bear_ob")):
                    ob_rows = '<div style="font-size:.82rem;color:var(--muted)">No active order blocks nearby.</div>'
                st.markdown(f'<div style="background:var(--card);border:1px solid var(--border);'
                            f'border-radius:10px;padding:1rem">{ob_rows}</div>',
                            unsafe_allow_html=True)

            colC, colD = st.columns(2)

            # Liquidity panel
            with colC:
                st.markdown('<div style="font-size:.85rem;font-weight:800;color:var(--text);'
                            'margin:.8rem 0 .5rem">💧 Liquidity Pools</div>',
                            unsafe_allow_html=True)
                nbs = ind.get("smc_nearest_buyside")
                nss = ind.get("smc_nearest_sellside")
                liq_rows = ""
                if nbs:
                    liq_rows += (f'<div style="font-size:.82rem;margin-bottom:.3rem">'
                                 f'🔼 Buy-side liquidity above: <b>${nbs["level"]}</b> '
                                 f'({nbs["touches"]} equal highs — short stops)</div>')
                if nss:
                    liq_rows += (f'<div style="font-size:.82rem;margin-bottom:.3rem">'
                                 f'🔽 Sell-side liquidity below: <b>${nss["level"]}</b> '
                                 f'({nss["touches"]} equal lows — long stops)</div>')
                if not (nbs or nss):
                    liq_rows = '<div style="font-size:.82rem;color:var(--muted)">No clear liquidity clusters nearby.</div>'
                st.markdown(f'<div style="background:var(--card);border:1px solid var(--border);'
                            f'border-radius:10px;padding:1rem">{liq_rows}</div>',
                            unsafe_allow_html=True)

            # How to read panel
            with colD:
                st.markdown('<div style="font-size:.85rem;font-weight:800;color:var(--text);'
                            'margin:.8rem 0 .5rem">📖 How to Read This</div>',
                            unsafe_allow_html=True)
                st.markdown(
                    '<div style="background:var(--card);border:1px solid var(--border);'
                    'border-radius:10px;padding:1rem;font-size:.78rem;color:var(--muted);'
                    'line-height:1.7">'
                    '<b style="color:var(--text)">Confluence is key:</b> a bullish setup is '
                    'strongest when price is in <b>Discount</b>, sitting at a <b>bull Order Block</b> '
                    'or <b>FVG</b>, with recent <b>bullish Displacement</b>. '
                    'Liquidity pools show where price is likely drawn next (stop hunts).'
                    '</div>',
                    unsafe_allow_html=True)

            # ── Confluence with existing signals ───────────────────────────────
            st.markdown('<div style="font-size:.85rem;font-weight:800;color:var(--text);'
                        'margin:1.2rem 0 .5rem">🔗 Confluence with Technical Signals</div>',
                        unsafe_allow_html=True)
            conf_items = []
            if ind.get("bull_trap"):
                conf_items.append(("🪤 Bull Trap active", "bear"))
            if ind.get("bear_trap"):
                conf_items.append(("🪤 Bear Trap active", "bull"))
            if ind.get("supertrend_bullish"): conf_items.append(("Supertrend Bullish", "bull"))
            else: conf_items.append(("Supertrend Bearish", "bear"))
            if ind.get("rsi"):
                if ind["rsi"] >= 70: conf_items.append((f"RSI Overbought ({ind['rsi']})", "bear"))
                elif ind["rsi"] <= 30: conf_items.append((f"RSI Oversold ({ind['rsi']})", "bull"))
            if score >= 35: conf_items.append(("SMC Bullish Confluence", "bull"))
            elif score <= -35: conf_items.append(("SMC Bearish Confluence", "bear"))

            chips = ""
            for txt, side in conf_items:
                c = theme_t["green"] if side == "bull" else theme_t["red"]
                chips += (f'<span style="background:rgba(0,0,0,.12);border:1px solid {c};'
                          f'color:{c};border-radius:6px;padding:.3rem .7rem;font-size:.78rem;'
                          f'font-weight:700;margin:.2rem">{txt}</span> ')
            st.markdown(f'<div style="display:flex;flex-wrap:wrap;gap:.3rem">{chips}</div>',
                        unsafe_allow_html=True)

    # ── Universe-wide SMC setup scanner ────────────────────────────────────────
    st.markdown("<hr style='border-color:var(--border);margin:1.5rem 0'>",
                unsafe_allow_html=True)
    st.markdown('<div class="sec">🎯 Scan Universe for SMC Setups</div>',
                unsafe_allow_html=True)

    if not _SMC_SCANNER_AVAILABLE:
        st.warning("Universe SMC scan requires the updated signals.py (with "
                   "scan_for_smc_setups). Deploy the latest signals.py to enable.",
                   icon="⚠️")
    else:
        scs1, scs2, scs3 = st.columns([1.2, 1.2, 1])
        with scs1:
            min_q = st.selectbox("Min quality", ["B", "A", "A+"],
                                 label_visibility="collapsed")
        with scs2:
            act_f = st.selectbox("Action", ["All", "BUY", "SELL"],
                                 label_visibility="collapsed")
        with scs3:
            run_smc_scan = st.button("🎯 Scan Setups", width="stretch")

        if run_smc_scan:
            with st.spinner(f"Scanning {min(MAX_SCAN_SYMBOLS, len(SECTOR_MAP))} stocks for SMC setups…"):
                st.session_state.smc_scan_cache = scan_for_smc_setups(
                    min_quality=min_q, action_filter=act_f)
                sc = st.session_state.smc_scan_cache
                st.toast(f"✅ {sc['buy_count']} BUY · {sc['sell_count']} SELL setups",
                         icon="🎯")

        sc = st.session_state.get("smc_scan_cache")
        if sc:
            st.markdown(
                f'<div style="font-size:.75rem;color:var(--muted);margin:.5rem 0">'
                f'Scanned {sc["scanned"]} · {sc["liquid"]} liquid · '
                f'{sc["buy_count"]} BUY · {sc["sell_count"]} SELL · {sc["timestamp"]}</div>',
                unsafe_allow_html=True)

            all_setups = sc["buy_setups"] + sc["sell_setups"]
            if all_setups:
                rows = []
                for s in all_setups:
                    rows.append({
                        "Stock": s["stock"], "Sector": s["sector"],
                        "Action": s["action"], "Grade": s["quality"],
                        "CMP": s["cmp"], "Entry": s["entry"],
                        "Target": s["target"], "SL": s["stop_loss"],
                        "RR": s["risk_reward"], "Zone": s["zone"],
                        "SMC": s["smc_score"],
                    })
                setup_df = pd.DataFrame(rows)
                # Dynamic height: ~35px per row + header, capped so it scrolls
                _dyn_h = min(max(len(setup_df) * 36 + 40, 200), 600)
                st.dataframe(
                    setup_df, hide_index=True, height=_dyn_h,
                    use_container_width=True, row_height=35,
                    column_config={
                        "Stock":  st.column_config.TextColumn("Stock", width="small", pinned=True),
                        "Action": st.column_config.TextColumn("Action", width="small"),
                        "Grade":  st.column_config.TextColumn("Grade", width="small"),
                        "CMP":    st.column_config.NumberColumn("CMP", format="$%.2f"),
                        "Entry":  st.column_config.NumberColumn("Entry", format="$%.2f"),
                        "Target": st.column_config.NumberColumn("Target", format="$%.2f"),
                        "SL":     st.column_config.NumberColumn("SL", format="$%.2f"),
                        "RR":     st.column_config.NumberColumn("R:R", format="%.2f"),
                        "SMC":    st.column_config.NumberColumn("Score", format="%d"),
                    })
                st.download_button(
                    "⬇️ Export SMC Setups CSV",
                    setup_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"smc_setups_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv")
            else:
                st.info("No setups found at this quality/action filter. Try lowering to grade B or 'All' actions.")
        else:
            st.info("Click **🎯 Scan Setups** to find SMC trade setups across the universe.")

# ── ETF Tracker ────────────────────────────────────────────────────────────────
elif _page == 'etfs':
    st.markdown('<div class="sec">📈 ETF Tracker</div>', unsafe_allow_html=True)
    if not _FUNDS_AVAILABLE:
        st.warning("⚠️ ETF/Fund module not available. Make sure `funds.py` is in the repo.")
    else:
        st.caption("NSE-listed ETFs — tracked separately from your stock portfolio. "
                   "Data via Yahoo Finance (15-min delayed).")

        # Session cache for ETF scan
        if "etf_scan_cache" not in st.session_state:
            st.session_state.etf_scan_cache = None

        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            if st.button("🔍 Scan All ETFs", width="stretch"):
                with st.spinner("Fetching ETF data…"):
                    st.session_state.etf_scan_cache = _funds.scan_etfs()
        with c2:
            _cat = st.selectbox("Category", ["All", "Equity Index", "Sectoral",
                                "Gold/Silver", "Debt/Liquid", "International"],
                                label_visibility="collapsed")

        # Single ETF lookup
        st.markdown("##### 🔎 Look up a specific ETF")
        etf_names = [f"{sym} — {name}" for sym, name in _funds.ETF_UNIVERSE.items()]
        sel = st.selectbox("Select ETF", etf_names, label_visibility="collapsed")
        sel_sym = sel.split(" — ")[0]
        if st.button("📊 Get ETF Details"):
            with st.spinner(f"Fetching {sel_sym}…"):
                q = _funds.get_etf_quote(sel_sym)
            if not q.get("has_data"):
                st.error(f"Couldn't fetch data for {sel_sym}. Yahoo may be rate-limited — try again.")
            else:
                r = q["returns"]
                day_clr = "var(--green)" if (q["day_chg"] or 0) >= 0 else "var(--red)"
                # Signal badge styling
                _sig = q.get("signal", "—")
                _sig_clr = {"BUY": "#10b981", "SELL": "#ef4444",
                            "HOLD": "#f59e0b"}.get(_sig, "#8e8e93")
                _sig_bg  = {"BUY": "rgba(16,185,129,.15)", "SELL": "rgba(239,68,68,.15)",
                            "HOLD": "rgba(245,158,11,.15)"}.get(_sig, "rgba(142,142,147,.15)")
                st.markdown(f"""
<div style="background:var(--card);border:1px solid var(--border);border-radius:12px;
            padding:1.2rem 1.5rem;margin:.5rem 0">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:.5rem">
    <div>
      <div style="font-size:1.1rem;font-weight:800;color:var(--accent)">{q['name']}</div>
      <div style="font-size:.8rem;color:var(--muted)">{sel_sym}</div>
    </div>
    <div style="text-align:right">
      <span style="background:{_sig_bg};color:{_sig_clr};font-weight:800;font-size:1rem;
                   padding:.35rem .9rem;border-radius:8px;border:1px solid {_sig_clr}">
            {_sig}</span>
      <div style="font-size:.7rem;color:var(--muted);margin-top:.25rem">
           Score {q.get('signal_score',0)} · {q.get('signal_confidence','—')} conf · RSI {q.get('rsi','—')}</div>
    </div>
  </div>
  <div style="display:flex;gap:2rem;flex-wrap:wrap;margin-top:.8rem">
    <div><span style="color:var(--muted);font-size:.75rem">CMP</span><br>
         <b style="font-size:1.3rem">${q['cmp']}</b></div>
    <div><span style="color:var(--muted);font-size:.75rem">Day</span><br>
         <b style="font-size:1.3rem;color:{day_clr}">{(q['day_chg'] or 0):+.2f}%</b></div>
    <div><span style="color:var(--muted);font-size:.75rem">50-DMA</span><br>
         <b style="font-size:1.1rem">${q.get('dma50','—')}</b></div>
    <div><span style="color:var(--muted);font-size:.75rem">200-DMA</span><br>
         <b style="font-size:1.1rem">${q.get('dma200') or '—'}</b></div>
    <div><span style="color:var(--muted);font-size:.75rem">52W High</span><br>
         <b style="font-size:1.1rem">${q['high52']}</b></div>
    <div><span style="color:var(--muted);font-size:.75rem">52W Low</span><br>
         <b style="font-size:1.1rem">${q['low52']}</b></div>
  </div>
  <div style="margin-top:1rem;display:flex;gap:1.5rem;flex-wrap:wrap">
    {''.join(f'<div><span style="color:var(--muted);font-size:.72rem">{k}</span><br>'
             f'<b style="color:{"var(--green)" if (v or 0)>=0 else "var(--red)"}">'
             f'{("+" if (v or 0)>=0 else "")}{v if v is not None else "—"}'
             f'{"%" if v is not None else ""}</b></div>'
             for k, v in r.items())}
  </div>
  {('<div style="margin-top:1rem;padding-top:.8rem;border-top:1px solid var(--border)">'
    '<span style="color:var(--muted);font-size:.72rem">📋 Why this signal:</span><br>'
    '<span style="font-size:.82rem">' + ' · '.join(q.get('signal_reasons', [])) + '</span></div>')
   if q.get('signal_reasons') else ''}
  {('<div style="margin-top:.8rem;font-size:.85rem">'
    f'<b>Entry</b> ${q.get("entry")} &nbsp; <b>Target</b> ${q.get("target")} &nbsp; '
    f'<b>Stop</b> ${q.get("stop_loss")}</div>')
   if q.get('signal')=='BUY' and q.get('target') else ''}
</div>""", unsafe_allow_html=True)
                st.caption("⚠️ Signals are algorithmic (trend + momentum), not financial advice. "
                           "ETFs carry market risk — do your own research.")

        # Full scan results
        if st.session_state.etf_scan_cache is not None:
            edf = st.session_state.etf_scan_cache
            if edf is not None and not edf.empty:
                st.markdown(f"##### 📋 All ETFs ({len(edf)}) — sorted by 1Y return")
                _h = min(max(len(edf) * 36 + 40, 200), 600)
                st.dataframe(edf, width="stretch", height=_h, hide_index=True,
                             column_config={"Symbol": st.column_config.TextColumn(pinned=True)})
                st.download_button(
                    "⬇️ Export ETF Data CSV",
                    edf.to_csv(index=False).encode("utf-8"),
                    file_name=f"etfs_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv")
            else:
                st.info("No ETF data returned. Yahoo Finance may be rate-limited — try again shortly.")
        else:
            st.info("Click **🔍 Scan All ETFs** to load the full ETF list with returns.")

# ── Mutual Fund Tracker ────────────────────────────────────────────────────────
elif _page == 'mutual_funds':
    st.markdown('<div class="sec">🏛 Mutual Fund Tracker</div>', unsafe_allow_html=True)
    if not _FUNDS_AVAILABLE:
        st.warning("⚠️ ETF/Fund module not available. Make sure `funds.py` is in the repo.")
    else:
        st.caption("Search any Indian mutual fund by name. NAV & returns via AMFI "
                   "(official, updated daily). Kept separate from your stock portfolio.")

        # Session state for MF
        for k, v in [("mf_search_results", []), ("mf_selected", None),
                     ("mf_compare_list", [])]:
            if k not in st.session_state:
                st.session_state[k] = v

        tab_search, tab_compare = st.tabs(["🔎 Search & Track", "⚖️ Compare Funds"])

        with tab_search:
            q = st.text_input("Search mutual fund",
                              placeholder="e.g. 'parag parikh flexi cap' or 'hdfc small cap'",
                              label_visibility="collapsed")
            if st.button("🔎 Search") and q:
                with st.spinner("Searching AMFI database…"):
                    st.session_state.mf_search_results = _funds.search_mf(q)
                if not st.session_state.mf_search_results:
                    st.warning("No funds found (need 3+ characters). Try the fund house + scheme name.")

            results = st.session_state.mf_search_results
            if results:
                opts = [f"{r['schemeName']}" for r in results]
                pick = st.selectbox(f"Found {len(results)} funds — select one:", opts)
                pick_code = next((r["schemeCode"] for r in results
                                  if r["schemeName"] == pick), None)

                cc1, cc2 = st.columns([1, 1])
                with cc1:
                    if st.button("📊 View Fund Details", width="stretch"):
                        st.session_state.mf_selected = pick_code
                with cc2:
                    if st.button("➕ Add to Compare", width="stretch"):
                        if pick_code and pick_code not in st.session_state.mf_compare_list:
                            st.session_state.mf_compare_list.append(pick_code)
                            st.toast(f"Added to compare ({len(st.session_state.mf_compare_list)})")

            # Render selected fund card
            if st.session_state.mf_selected:
                with st.spinner("Loading fund data…"):
                    s = _funds.mf_summary(st.session_state.mf_selected)
                if not s.get("has_history"):
                    st.error("Couldn't load this fund's NAV history. Try again shortly.")
                else:
                    r = s["returns"]
                    _rt = s.get("rating", {})
                    _rt_clr = {"STRONG": "#10b981", "GOOD": "#22c55e",
                               "AVERAGE": "#f59e0b", "WEAK": "#f97316",
                               "POOR": "#ef4444"}.get(_rt.get("rating"), "#8e8e93")
                    _rt_bg = {"STRONG": "rgba(16,185,129,.15)", "GOOD": "rgba(34,197,94,.12)",
                              "AVERAGE": "rgba(245,158,11,.15)", "WEAK": "rgba(249,115,22,.15)",
                              "POOR": "rgba(239,68,68,.15)"}.get(_rt.get("rating"), "rgba(142,142,147,.15)")
                    st.markdown(f"""
<div style="background:var(--card);border:1px solid var(--border);border-radius:12px;
            padding:1.2rem 1.5rem;margin:.5rem 0">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:.5rem">
    <div>
      <div style="font-size:1.05rem;font-weight:800;color:var(--accent)">{s['scheme_name']}</div>
      <div style="font-size:.78rem;color:var(--muted)">
           {s.get('fund_house','')} · {s.get('scheme_category','')}</div>
    </div>
    <div style="text-align:right">
      <span style="background:{_rt_bg};color:{_rt_clr};font-weight:800;font-size:.95rem;
                   padding:.35rem .9rem;border-radius:8px;border:1px solid {_rt_clr}">
            {'⭐'*_rt.get('stars',0)} {_rt.get('rating','—')}</span>
      <div style="font-size:.7rem;color:var(--muted);margin-top:.25rem">Quality score {_rt.get('score',0)}/100</div>
    </div>
  </div>
  <div style="display:flex;gap:2rem;flex-wrap:wrap;margin-top:.8rem">
    <div><span style="color:var(--muted);font-size:.75rem">NAV</span><br>
         <b style="font-size:1.3rem">${s['nav']}</b></div>
    <div><span style="color:var(--muted);font-size:.75rem">As of</span><br>
         <b style="font-size:1rem">{s['nav_date']}</b></div>
    <div><span style="color:var(--muted);font-size:.75rem">52W High</span><br>
         <b style="font-size:1.1rem">${s['high52']}</b></div>
    <div><span style="color:var(--muted);font-size:.75rem">52W Low</span><br>
         <b style="font-size:1.1rem">${s['low52']}</b></div>
  </div>
  <div style="margin-top:1rem;display:flex;gap:1.5rem;flex-wrap:wrap">
    {''.join(f'<div><span style="color:var(--muted);font-size:.72rem">{k}{" (CAGR)" if k in ("3Y","5Y") else ""}</span><br>'
             f'<b style="color:{"var(--green)" if (v or 0)>=0 else "var(--red)"}">'
             f'{("+" if (v or 0)>=0 else "")}{v if v is not None else "—"}'
             f'{"%" if v is not None else ""}</b></div>'
             for k, v in r.items())}
  </div>
  <div style="margin-top:1rem;padding-top:.8rem;border-top:1px solid var(--border);font-size:.85rem">
       💡 <b>{_rt.get('action','—')}</b></div>
</div>""", unsafe_allow_html=True)
                    st.caption("⚠️ Ratings reflect past performance (returns + consistency), "
                               "not a buy/sell timing call. Mutual funds are long-term "
                               "instruments — past returns don't guarantee future results.")

                    # NAV history chart
                    hist = _funds.get_mf_history(st.session_state.mf_selected)
                    if hist is not None and not hist.empty:
                        chart_df = hist.set_index("date")["nav"].tail(365)
                        st.line_chart(chart_df, height=260)

        with tab_compare:
            clist = st.session_state.mf_compare_list
            if not clist:
                st.info("Add funds via the **Search & Track** tab → '➕ Add to Compare'.")
            else:
                st.markdown(f"##### Comparing {len(clist)} funds")
                if st.button("🗑 Clear comparison"):
                    st.session_state.mf_compare_list = []
                    st.rerun()
                with st.spinner("Building comparison…"):
                    cmp_df = _funds.compare_mfs(clist)
                if cmp_df is not None and not cmp_df.empty:
                    st.dataframe(cmp_df, width="stretch", hide_index=True,
                                 column_config={"Fund": st.column_config.TextColumn(pinned=True)})
                    st.download_button(
                        "⬇️ Export Comparison CSV",
                        cmp_df.to_csv(index=False).encode("utf-8"),
                        file_name=f"mf_compare_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv")

# ── Position Sizing Calculator ─────────────────────────────────────────────────
elif _page == 'sizing':
    st.markdown('<div class="sec">🧮 Position Sizing Calculator</div>',
                unsafe_allow_html=True)
    st.caption("Risk-based position sizing — never lose more than your chosen % "
               "on a single trade. This is the most important discipline in trading.")

    colA, colB = st.columns(2)
    with colA:
        _cap = st.number_input("💰 Total Trading Capital ($)", min_value=1000.0,
                               value=float(st.session_state.get("_sz_cap", 100000.0)),
                               step=5000.0, key="sz_cap")
        st.session_state._sz_cap = _cap
        _risk_pct = st.slider("⚠️ Risk per Trade (%)", 0.25, 5.0,
                              float(st.session_state.get("_sz_risk", 1.0)), 0.25,
                              key="sz_risk",
                              help="Pros risk 1-2% per trade. Never exceed 2% as a beginner.")
        st.session_state._sz_risk = _risk_pct
    with colB:
        _entry = st.number_input("📈 Entry Price ($)", min_value=0.0,
                                 value=float(st.session_state.get("_sz_entry", 100.0)),
                                 step=1.0, key="sz_entry")
        st.session_state._sz_entry = _entry
        _stop = st.number_input("🛑 Stop Loss Price ($)", min_value=0.0,
                                value=float(st.session_state.get("_sz_stop", 95.0)),
                                step=1.0, key="sz_stop")
        st.session_state._sz_stop = _stop

    # Calculations
    if _entry > 0 and _stop > 0 and _entry != _stop:
        risk_amount = _cap * (_risk_pct / 100)          # max $ to lose
        risk_per_share = abs(_entry - _stop)             # $ risk per share
        qty = int(risk_amount / risk_per_share) if risk_per_share > 0 else 0
        position_value = qty * _entry
        position_pct = (position_value / _cap * 100) if _cap > 0 else 0
        is_long = _stop < _entry
        direction = "LONG" if is_long else "SHORT"

        # Warnings
        warn = ""
        if position_value > _cap:
            warn = ("⚠️ This position needs more capital than you have — "
                    "your stop is too wide for this risk %. Widen the stop "
                    "distance or lower risk %.")
        elif position_pct > 50:
            warn = (f"⚠️ This position is {position_pct:.0f}% of your capital — "
                    "very concentrated. Consider a tighter stop or smaller risk %.")

        st.markdown(f"""
<div style="background:var(--card);border:1px solid var(--border);border-radius:12px;
            padding:1.4rem 1.6rem;margin:1rem 0">
  <div style="font-size:.75rem;color:var(--muted);text-transform:uppercase;
              letter-spacing:.08em;margin-bottom:.8rem">{direction} Position</div>
  <div style="display:flex;gap:2.5rem;flex-wrap:wrap">
    <div><span style="color:var(--muted);font-size:.78rem">Shares to Buy</span><br>
         <b style="font-size:1.8rem;color:var(--accent)">{qty:,}</b></div>
    <div><span style="color:var(--muted);font-size:.78rem">Position Value</span><br>
         <b style="font-size:1.5rem">${position_value:,.0f}</b>
         <span style="font-size:.8rem;color:var(--muted)"> ({position_pct:.1f}%)</span></div>
    <div><span style="color:var(--muted);font-size:.78rem">Max Loss (your risk)</span><br>
         <b style="font-size:1.5rem;color:var(--red)">${risk_amount:,.0f}</b></div>
    <div><span style="color:var(--muted);font-size:.78rem">Risk / Share</span><br>
         <b style="font-size:1.5rem">${risk_per_share:.2f}</b></div>
  </div>
</div>""", unsafe_allow_html=True)
        if warn:
            st.warning(warn)

        # Target calculator with R-multiples
        st.markdown("##### 🎯 Target Levels (Risk:Reward)")
        tcols = st.columns(4)
        for i, rmult in enumerate([1, 2, 3, 5]):
            if is_long:
                tgt = _entry + rmult * risk_per_share
            else:
                tgt = _entry - rmult * risk_per_share
            reward = qty * abs(tgt - _entry)
            with tcols[i]:
                st.markdown(f"""
<div style="background:var(--card);border:1px solid var(--border);border-radius:8px;
            padding:.7rem;text-align:center">
  <div style="font-size:.7rem;color:var(--muted)">{rmult}R Target</div>
  <div style="font-size:1.1rem;font-weight:800;color:var(--green)">${tgt:.2f}</div>
  <div style="font-size:.72rem;color:var(--muted)">+${reward:,.0f}</div>
</div>""", unsafe_allow_html=True)

        st.caption("💡 Quantity is calculated so that IF your stop loss hits, you lose "
                   "exactly your chosen risk amount — no more. This is how professionals "
                   "size every trade.")
    else:
        st.info("Enter entry and stop loss prices (they must differ) to calculate sizing.")

# ── Risk Dashboard ─────────────────────────────────────────────────────────────
elif _page == 'risk':
    st.markdown('<div class="sec">🛡 Portfolio Risk Dashboard</div>',
                unsafe_allow_html=True)
    st.caption("Portfolio-level risk exposure across all open positions.")

    odf_risk = df[df["status"] == "Open"].copy() if not df.empty else pd.DataFrame()
    if odf_risk.empty:
        st.info("No open positions to analyze. Add trades to see your risk profile.")
    else:
        total_cap = st.number_input("💰 Total Trading Capital ($) — for risk context",
                                    min_value=1000.0,
                                    value=float(st.session_state.get("_sz_cap", 100000.0)),
                                    step=5000.0, key="risk_cap")
        st.session_state._sz_cap = total_cap

        total_invested = odf_risk["invested"].sum()
        total_current  = odf_risk["current_amt"].sum()
        n_positions    = len(odf_risk)

        # Concentration: largest position
        odf_risk["pos_value"] = odf_risk["current_amt"]
        largest = odf_risk.loc[odf_risk["pos_value"].idxmax()]
        largest_pct = (largest["pos_value"] / total_current * 100) if total_current > 0 else 0

        # Capital deployed
        deployed_pct = (total_invested / total_cap * 100) if total_cap > 0 else 0

        # Sector concentration
        odf_risk["sector"] = odf_risk["stock"].apply(get_sector)
        sector_exposure = odf_risk.groupby("sector")["pos_value"].sum()
        top_sector = sector_exposure.idxmax() if not sector_exposure.empty else "—"
        top_sector_pct = (sector_exposure.max() / total_current * 100) if total_current > 0 else 0

        # Risk badges
        def _risk_badge(value, warn_thresh, danger_thresh, reverse=False):
            if reverse:
                lvl = "danger" if value < danger_thresh else "warn" if value < warn_thresh else "ok"
            else:
                lvl = "danger" if value > danger_thresh else "warn" if value > warn_thresh else "ok"
            clr = {"ok": "#10b981", "warn": "#f59e0b", "danger": "#ef4444"}[lvl]
            return clr

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            clr = _risk_badge(deployed_pct, 80, 95)
            st.markdown(f"""<div style="background:var(--card);border:1px solid var(--border);
                border-radius:10px;padding:1rem;text-align:center">
                <div style="font-size:.72rem;color:var(--muted)">Capital Deployed</div>
                <div style="font-size:1.6rem;font-weight:800;color:{clr}">{deployed_pct:.0f}%</div>
                <div style="font-size:.7rem;color:var(--muted)">${total_invested:,.0f}</div>
                </div>""", unsafe_allow_html=True)
        with m2:
            clr = _risk_badge(largest_pct, 25, 40)
            st.markdown(f"""<div style="background:var(--card);border:1px solid var(--border);
                border-radius:10px;padding:1rem;text-align:center">
                <div style="font-size:.72rem;color:var(--muted)">Largest Position</div>
                <div style="font-size:1.6rem;font-weight:800;color:{clr}">{largest_pct:.0f}%</div>
                <div style="font-size:.7rem;color:var(--muted)">{largest['stock']}</div>
                </div>""", unsafe_allow_html=True)
        with m3:
            clr = _risk_badge(top_sector_pct, 40, 60)
            st.markdown(f"""<div style="background:var(--card);border:1px solid var(--border);
                border-radius:10px;padding:1rem;text-align:center">
                <div style="font-size:.72rem;color:var(--muted)">Top Sector</div>
                <div style="font-size:1.6rem;font-weight:800;color:{clr}">{top_sector_pct:.0f}%</div>
                <div style="font-size:.7rem;color:var(--muted)">{top_sector}</div>
                </div>""", unsafe_allow_html=True)
        with m4:
            st.markdown(f"""<div style="background:var(--card);border:1px solid var(--border);
                border-radius:10px;padding:1rem;text-align:center">
                <div style="font-size:.72rem;color:var(--muted)">Open Positions</div>
                <div style="font-size:1.6rem;font-weight:800;color:var(--accent)">{n_positions}</div>
                <div style="font-size:.7rem;color:var(--muted)">holdings</div>
                </div>""", unsafe_allow_html=True)

        # Sector exposure breakdown
        st.markdown("##### 🥧 Sector Exposure")
        sec_df = (sector_exposure / total_current * 100).round(1).reset_index()
        sec_df.columns = ["Sector", "% of Portfolio"]
        sec_df = sec_df.sort_values("% of Portfolio", ascending=False)
        st.dataframe(sec_df, width="stretch", hide_index=True)

        # Position-level risk table
        st.markdown("##### 📊 Position Breakdown")
        pos_df = odf_risk[["stock", "invested", "current_amt", "profit", "profit_pct"]].copy()
        pos_df["% of Portfolio"] = (pos_df["current_amt"] / total_current * 100).round(1)
        pos_df = pos_df.rename(columns={"stock": "Stock", "invested": "Invested",
                                        "current_amt": "Current", "profit": "P&L",
                                        "profit_pct": "P&L %"})
        pos_df = pos_df.sort_values("% of Portfolio", ascending=False)
        st.dataframe(pos_df, width="stretch", hide_index=True)

        # Health summary
        st.markdown("##### 🩺 Risk Health Check")
        issues = []
        if deployed_pct > 95:
            issues.append("🔴 Nearly fully invested — no dry powder for opportunities or averaging.")
        if largest_pct > 40:
            issues.append(f"🔴 {largest['stock']} is {largest_pct:.0f}% of your portfolio — "
                          "a single stock shock could badly hurt you.")
        if top_sector_pct > 60:
            issues.append(f"🔴 {top_sector} sector is {top_sector_pct:.0f}% of holdings — "
                          "heavy concentration risk if that sector falls.")
        if not issues:
            st.success("✅ Portfolio risk looks reasonably balanced — no major concentration flags.")
        else:
            for i in issues:
                st.warning(i)

# ── Price Alerts ───────────────────────────────────────────────────────────────
elif _page == 'alerts':
    st.markdown('<div class="sec">🔔 Price Alerts</div>', unsafe_allow_html=True)
    st.caption("Set price targets on any stock. Alerts trigger when crossed and "
               "(if Telegram is configured) send you a message.")

    with st.expander("➕ Create New Alert", expanded=True):
        ac1, ac2, ac3 = st.columns([2, 1, 1])
        with ac1:
            _al_stock = st.text_input("Stock symbol (NSE)", key="al_stock",
                                      placeholder="e.g. RELIANCE")
        with ac2:
            _al_cond = st.selectbox("Condition", ["above", "below"], key="al_cond")
        with ac3:
            _al_price = st.number_input("Target $", min_value=0.0, step=1.0, key="al_price")
        _al_note = st.text_input("Note (optional)", key="al_note",
                                 placeholder="e.g. breakout level / support")
        if st.button("🔔 Create Alert", width="stretch"):
            if _al_stock and _al_price > 0:
                add_price_alert(UID, _al_stock, _al_cond, _al_price, _al_note)
                st.success(f"Alert set: {_al_stock.upper()} {_al_cond} ${_al_price}")
                st.rerun()
            else:
                st.error("Enter a stock symbol and a target price above 0.")

    # Active alerts — check against live prices
    active = get_price_alerts(UID, status="Active")
    if active:
        st.markdown(f"##### 🟢 Active Alerts ({len(active)})")
        # Fetch current prices for all alert stocks
        alert_syms = tuple(sorted({a[1] for a in active}))
        alert_prices = _cached_prices(alert_syms)

        for a in active:
            aid, stock, cond, target, status, note, created, _ = a
            cur_price = alert_prices.get(stock)
            triggered = False
            if cur_price is not None:
                if cond == "above" and cur_price >= target:
                    triggered = True
                elif cond == "below" and cur_price <= target:
                    triggered = True

            cur_str = f"${cur_price}" if cur_price is not None else "—"
            arrow = "▲" if cond == "above" else "▼"
            row_clr = "#10b981" if triggered else "var(--border)"

            cc1, cc2 = st.columns([5, 1])
            with cc1:
                trig_html = ('<span style="background:rgba(16,185,129,.2);color:#10b981;'
                             'padding:.15rem .5rem;border-radius:4px;font-size:.7rem;'
                             'font-weight:800;margin-left:.5rem">🎯 TRIGGERED</span>') if triggered else ''
                st.markdown(f"""
<div style="background:var(--card);border:1px solid {row_clr};border-radius:8px;
            padding:.7rem 1rem;margin-bottom:.5rem">
  <b style="font-size:.95rem">{stock}</b> {arrow} ${target}
  <span style="color:var(--muted);font-size:.8rem">· now {cur_str}</span>{trig_html}
  {('<br><span style="font-size:.72rem;color:var(--muted)">📝 ' + note + '</span>') if note else ''}
</div>""", unsafe_allow_html=True)
            with cc2:
                if st.button("🗑", key=f"del_alert_{aid}"):
                    delete_price_alert(aid, UID)
                    st.rerun()

            # If triggered, mark it + optionally send Telegram
            if triggered:
                trigger_price_alert(aid, UID)
                if saved_tok and saved_cid:
                    try:
                        send_telegram(saved_tok, saved_cid,
                            f"🎯 <b>PRICE ALERT</b>\n{stock} is now {cur_str} "
                            f"({arrow} target ${target})\n{note}")
                    except Exception:
                        pass
    else:
        st.info("No active alerts. Create one above.")

    # Triggered history
    triggered_alerts = get_price_alerts(UID, status="Triggered")
    if triggered_alerts:
        with st.expander(f"📜 Triggered History ({len(triggered_alerts)})"):
            for a in triggered_alerts:
                aid, stock, cond, target, status, note, created, trig_date = a
                arrow = "▲" if cond == "above" else "▼"
                tc1, tc2 = st.columns([5, 1])
                with tc1:
                    st.markdown(f"**{stock}** {arrow} ${target} · triggered {trig_date or '—'}")
                with tc2:
                    if st.button("🗑", key=f"del_trig_{aid}"):
                        delete_price_alert(aid, UID)
                        st.rerun()

# ── Stock Candlestick Chart ────────────────────────────────────────────────────
elif _page == 'chart':
    st.markdown('<div class="sec">📈 Stock Chart</div>', unsafe_allow_html=True)
    st.caption("Candlestick chart with EMAs, Bollinger Bands, and support/resistance. "
               "Pick a holding, watchlist stock, or type any NSE symbol.")

    # Build symbol options: holdings + watchlist + manual
    hold_syms = sorted(df["stock"].unique().tolist()) if not df.empty else []
    try:
        wl = get_watchlist(UID)
        wl_syms = sorted(wl["stock"].unique().tolist()) if (wl is not None and not wl.empty) else []
    except Exception:
        wl_syms = []
    combined = sorted(set(hold_syms + wl_syms))

    cc1, cc2, cc3 = st.columns([2, 1, 1])
    with cc1:
        if combined:
            _picked = st.selectbox("Your stocks", ["— type below —"] + combined, key="chart_pick")
        else:
            _picked = "— type below —"
    with cc2:
        _typed = st.text_input("Or NSE symbol", key="chart_typed", placeholder="e.g. TCS")
    with cc3:
        _tf_label = st.selectbox("Timeframe", list(_CHART_TIMEFRAMES.keys()),
                                 index=3, key="chart_tf")
    _period, _interval = _CHART_TIMEFRAMES[_tf_label]

    chart_sym = (_typed.strip().upper() if _typed.strip()
                 else (_picked if _picked != "— type below —" else None))

    if not chart_sym:
        st.info("Select or type a stock symbol to view its chart.")
    else:
        with st.spinner(f"Loading {chart_sym} {_tf_label} chart…"):
            cdf = fetch_chart_data(chart_sym, _period, _interval)
            # Live CMP (real-time last price, same source as portfolio)
            live_cmp = None
            try:
                _lp = _cached_prices((chart_sym,))
                live_cmp = _lp.get(chart_sym)
            except Exception:
                pass

        if cdf is None or cdf.empty or len(cdf) < 5:
            st.error(f"Couldn't load {_tf_label} data for {chart_sym}. "
                     f"Intraday data may be unavailable (markets closed / newly listed), "
                     f"or Yahoo is rate-limited. Try the Daily timeframe.")
        else:
            c = cdf.copy()
            # Indicators for overlay
            close = c["Close"]
            _ema_fast = 20 if len(c) >= 20 else max(2, len(c) // 2)
            _ema_slow = 50 if len(c) >= 50 else max(3, len(c) // 2)
            c["EMA20"] = close.ewm(span=_ema_fast, adjust=False).mean()
            c["EMA50"] = close.ewm(span=_ema_slow, adjust=False).mean()
            _bb_win = min(20, len(c))
            bb_mid = close.rolling(_bb_win).mean()
            bb_std = close.rolling(_bb_win).std()
            c["BB_up"] = bb_mid + 2 * bb_std
            c["BB_dn"] = bb_mid - 2 * bb_std
            support = float(c["Low"].rolling(min(20, len(c))).min().iloc[-1])
            resistance = float(c["High"].rolling(min(20, len(c))).max().iloc[-1])

            fig = go.Figure()
            # Bollinger band fill
            fig.add_trace(go.Scatter(x=c.index, y=c["BB_up"], line=dict(width=0),
                                     showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=c.index, y=c["BB_dn"], line=dict(width=0),
                                     fill="tonexty", fillcolor="rgba(120,120,200,0.08)",
                                     showlegend=False, hoverinfo="skip", name="BB"))
            # Candles
            fig.add_trace(go.Candlestick(
                x=c.index, open=c["Open"], high=c["High"], low=c["Low"], close=c["Close"],
                name=chart_sym, increasing_line_color="#10b981",
                decreasing_line_color="#ef4444"))
            # EMAs
            fig.add_trace(go.Scatter(x=c.index, y=c["EMA20"],
                                     line=dict(color="#f59e0b", width=1.3),
                                     name=f"EMA {_ema_fast}"))
            fig.add_trace(go.Scatter(x=c.index, y=c["EMA50"],
                                     line=dict(color="#3b82f6", width=1.3),
                                     name=f"EMA {_ema_slow}"))
            # S/R lines
            fig.add_hline(y=resistance, line=dict(color="#ef4444", width=1, dash="dash"),
                          annotation_text=f"R ${resistance:.1f}", annotation_position="right")
            fig.add_hline(y=support, line=dict(color="#10b981", width=1, dash="dash"),
                          annotation_text=f"S ${support:.1f}", annotation_position="right")
            # Live CMP line (real-time, distinct from last candle close)
            if live_cmp:
                fig.add_hline(y=live_cmp, line=dict(color="#d4af37", width=1.2, dash="dot"),
                              annotation_text=f"CMP ${live_cmp:.1f}",
                              annotation_position="left")

            fig.update_layout(
                height=520, margin=dict(l=10, r=10, t=30, b=10),
                dragmode="pan",   # drag = pan (smooth), not the clunky zoom-box default
                xaxis_rangeslider_visible=True,   # bottom mini-slider for left/right scroll
                xaxis_rangeslider_thickness=0.06,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color=theme_t.get("text", "#fff")),
                legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
                hovermode="x unified")
            fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)", rangeslider_thickness=0.06)
            fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)", fixedrange=False)
            _chart_config = {
                "scrollZoom": True,        # mouse wheel / pinch zoom — smooth in/out
                "displaylogo": False,
                "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                "doubleClick": "autosize", # double-click resets zoom
            }
            st.plotly_chart(fig, use_container_width=True, config=_chart_config)

            # Volume chart below — x-axis matched to price chart range for sync scroll
            vfig = go.Figure()
            vol_clr = ["#10b981" if c["Close"].iloc[i] >= c["Open"].iloc[i]
                       else "#ef4444" for i in range(len(c))]
            vfig.add_trace(go.Bar(x=c.index, y=c["Volume"], marker_color=vol_clr,
                                  name="Volume"))
            vfig.update_layout(height=160, margin=dict(l=10, r=10, t=10, b=10),
                               dragmode="pan",
                               paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               font=dict(color=theme_t.get("text", "#fff")), showlegend=False)
            vfig.update_xaxes(gridcolor="rgba(255,255,255,0.05)",
                              range=[c.index[0], c.index[-1]])
            vfig.update_yaxes(gridcolor="rgba(255,255,255,0.05)")
            st.plotly_chart(vfig, use_container_width=True, config=_chart_config)
            st.caption("🖱️ Scroll to zoom · Drag to pan · Double-click to reset · "
                       "Use the slider strip below the chart to jump left/right")

            # Quick stats row — lead with LIVE CMP, not stale candle close
            last_close = float(c["Close"].iloc[-1])
            prev_close = float(c["Close"].iloc[-2]) if len(c) >= 2 else last_close
            display_price = live_cmp if live_cmp else last_close
            chg = (display_price / prev_close - 1) * 100 if prev_close else 0
            sc1, sc2, sc3, sc4 = st.columns(4)
            sc1.metric("CMP (live)" if live_cmp else "Last Close",
                       f"${display_price:.2f}", f"{chg:+.2f}%")
            sc2.metric("Support", f"${support:.2f}")
            sc3.metric("Resistance", f"${resistance:.2f}")
            sc4.metric("Bars", f"{len(c)}")
            if live_cmp and abs(live_cmp - last_close) > 0.01:
                st.caption(f"💡 Gold dotted line = live CMP ${live_cmp:.2f}. "
                           f"Last {_tf_label} candle closed at ${last_close:.2f} "
                           f"(candles lag live price, especially intraday/after-hours).")

# ── Trade Journal ──────────────────────────────────────────────────────────────
elif _page == 'journal':
    st.markdown('<div class="sec">📓 Trade Journal</div>', unsafe_allow_html=True)
    st.caption("Log why you entered each trade and what you learned. Over time this "
               "becomes your most valuable asset — you'll spot patterns in your own decisions.")

    with st.expander("➕ New Journal Entry", expanded=False):
        jc1, jc2, jc3 = st.columns(3)
        with jc1:
            j_stock = st.text_input("Stock", key="j_stock", placeholder="RELIANCE")
            j_date = st.date_input("Trade date", key="j_date")
            j_dir = st.selectbox("Direction", ["Long", "Short"], key="j_dir")
        with jc2:
            j_entry = st.number_input("Entry $", min_value=0.0, step=1.0, key="j_entry")
            j_exit = st.number_input("Exit $ (0 if open)", min_value=0.0, step=1.0, key="j_exit")
            j_setup = st.selectbox("Setup", ["Breakout", "Pullback", "Reversal", "Trend-follow",
                                   "SMC / Order Block", "Trap reversal", "Sector rotation",
                                   "News-based", "Other"], key="j_setup")
        with jc3:
            j_emotion = st.selectbox("Emotion at entry", ["Confident", "Neutral", "FOMO",
                                     "Fearful", "Greedy", "Revenge", "Disciplined"], key="j_emotion")
            j_outcome = st.selectbox("Outcome", ["Open", "Win", "Loss", "Breakeven"], key="j_outcome")
            j_rating = st.slider("Execution rating", 1, 5, 3, key="j_rating",
                                 help="How well did you follow your plan? (not P&L)")
        j_rationale = st.text_area("Why did you enter? (your thesis)", key="j_rationale",
                                   placeholder="e.g. broke 200-DMA on volume, sector rotating in…")
        j_lesson = st.text_area("Lesson learned / notes", key="j_lesson",
                                placeholder="e.g. exited too early out of fear, should have trusted the stop")
        if st.button("💾 Save Entry", width="stretch"):
            if j_stock:
                add_journal_entry(UID, j_stock, str(j_date), j_dir, j_entry,
                                  j_exit if j_exit > 0 else None, j_setup, j_rationale,
                                  j_emotion, j_outcome, j_lesson, j_rating)
                st.success(f"Journal entry saved for {j_stock.upper()}")
                st.rerun()
            else:
                st.error("Enter at least a stock symbol.")

    entries = get_journal_entries(UID)
    if not entries:
        st.info("No journal entries yet. Log your first trade above — your future self will thank you.")
    else:
        # Insight summary
        wins = sum(1 for e in entries if e[9] == "Win")
        losses = sum(1 for e in entries if e[9] == "Loss")
        closed = wins + losses
        avg_rating = sum(e[11] or 0 for e in entries) / len(entries) if entries else 0
        # Most common setup & emotion
        from collections import Counter
        setups = Counter(e[6] for e in entries if e[6])
        emotions = Counter(e[8] for e in entries if e[8])
        top_setup = setups.most_common(1)[0][0] if setups else "—"
        top_emotion = emotions.most_common(1)[0][0] if emotions else "—"

        ic1, ic2, ic3, ic4 = st.columns(4)
        ic1.metric("Entries", len(entries))
        ic2.metric("Win Rate", f"{(wins/closed*100):.0f}%" if closed else "—",
                   f"{wins}W / {losses}L")
        ic3.metric("Avg Execution", f"{avg_rating:.1f}/5")
        ic4.metric("Top Setup", top_setup)

        # Win rate by setup — the real insight
        st.markdown("##### 📊 Win Rate by Setup")
        setup_stats = {}
        for e in entries:
            s, oc = e[6], e[9]
            if oc in ("Win", "Loss"):
                setup_stats.setdefault(s, {"w": 0, "l": 0})
                setup_stats[s]["w" if oc == "Win" else "l"] += 1
        if setup_stats:
            rows = []
            for s, st_ in setup_stats.items():
                tot = st_["w"] + st_["l"]
                wr = st_["w"] / tot * 100 if tot else 0
                rows.append({"Setup": s, "Trades": tot, "Wins": st_["w"],
                             "Losses": st_["l"], "Win %": round(wr, 0)})
            sdf = pd.DataFrame(rows).sort_values("Win %", ascending=False)
            st.dataframe(sdf, width="stretch", hide_index=True)
            st.caption("💡 Focus on the setups where you actually win. Cut the ones that lose.")

        # Entry cards
        st.markdown("##### 📒 Entries")
        for e in entries:
            (jid, stock, tdate, direction, entry_p, exit_p, setup, rationale,
             emotion, outcome, lesson, rating, created) = e
            oc_clr = {"Win": "#10b981", "Loss": "#ef4444",
                      "Breakeven": "#f59e0b", "Open": "#8e8e93"}.get(outcome, "#8e8e93")
            pnl_str = ""
            if entry_p and exit_p:
                pnl = ((exit_p / entry_p - 1) * 100) if direction == "Long" else ((entry_p / exit_p - 1) * 100)
                pnl_str = f" · {pnl:+.1f}%"
            jcol1, jcol2 = st.columns([6, 1])
            with jcol1:
                st.markdown(f"""
<div style="background:var(--card);border:1px solid var(--border);border-left:3px solid {oc_clr};
            border-radius:8px;padding:.8rem 1rem;margin-bottom:.6rem">
  <div style="display:flex;justify-content:space-between;flex-wrap:wrap">
    <b style="font-size:.95rem">{stock} <span style="color:var(--muted);font-weight:400">
       {direction} · {setup}</span></b>
    <span style="color:{oc_clr};font-weight:700;font-size:.85rem">{outcome}{pnl_str}</span>
  </div>
  <div style="font-size:.78rem;color:var(--muted);margin-top:.2rem">
    {tdate} · Entry ${entry_p or '—'} → Exit ${exit_p or '—'} · Emotion: {emotion} · ⭐{rating}/5</div>
  {('<div style="font-size:.82rem;margin-top:.4rem"><b>Thesis:</b> ' + rationale + '</div>') if rationale else ''}
  {('<div style="font-size:.82rem;margin-top:.3rem;color:var(--accent)"><b>Lesson:</b> ' + lesson + '</div>') if lesson else ''}
</div>""", unsafe_allow_html=True)
            with jcol2:
                if st.button("🗑", key=f"del_journal_{jid}"):
                    delete_journal_entry(jid, UID)
                    st.rerun()

# ── Market Breadth ─────────────────────────────────────────────────────────────
elif _page == 'breadth':
    st.markdown('<div class="sec">📊 Market Breadth</div>', unsafe_allow_html=True)
    st.caption("Overall market health from the universe scan. Strong breadth = broad "
               "participation (safer for longs). Weak breadth = narrow/risky market.")

    scan_df = st.session_state.get("scanner_cache")
    if scan_df is None or (hasattr(scan_df, "empty") and scan_df.empty):
        st.info("Breadth needs universe scan data. Click below to run it "
                "(or it fills automatically via the background deep scan).")
        if st.button("🔍 Run Universe Scan for Breadth"):
            with st.spinner("Scanning universe…"):
                st.session_state.scanner_cache = generate_market_scanner()
            st.rerun()
    else:
        total = len(scan_df)
        # Breadth metrics from scanner columns
        uptrend = len(scan_df[scan_df["Trend"].isin(["Uptrend", "Strong Uptrend"])])
        downtrend = len(scan_df[scan_df["Trend"].isin(["Downtrend", "Strong Downtrend"])])
        bullish_pct = round(uptrend / total * 100, 1) if total else 0
        bearish_pct = round(downtrend / total * 100, 1) if total else 0
        # RSI-based
        overbought = len(scan_df[scan_df["RSI"] >= 70])
        oversold = len(scan_df[scan_df["RSI"] <= 30])
        # Signal distribution
        strong_buy = len(scan_df[scan_df["Signal"].str.contains("STRONG BUY", na=False)])
        avoid = len(scan_df[scan_df["Signal"].str.contains("AVOID", na=False)])

        # Overall breadth verdict
        if bullish_pct >= 60:
            verdict, vclr = "STRONG — broad participation, favorable for longs", "#10b981"
        elif bullish_pct >= 40:
            verdict, vclr = "NEUTRAL — mixed market, be selective", "#f59e0b"
        else:
            verdict, vclr = "WEAK — narrow market, longs carry extra risk", "#ef4444"

        st.markdown(f"""
<div style="background:var(--card);border:1px solid {vclr};border-radius:12px;
            padding:1.2rem 1.5rem;margin:.5rem 0">
  <div style="font-size:.75rem;color:var(--muted);text-transform:uppercase;
              letter-spacing:.08em">Market Breadth Verdict</div>
  <div style="font-size:1.2rem;font-weight:800;color:{vclr};margin-top:.3rem">{verdict}</div>
  <div style="font-size:.8rem;color:var(--muted);margin-top:.3rem">
       Based on {total:,} liquid stocks scanned</div>
</div>""", unsafe_allow_html=True)

        b1, b2, b3, b4 = st.columns(4)
        b1.metric("🟢 In Uptrend", f"{bullish_pct}%", f"{uptrend} stocks")
        b2.metric("🔴 In Downtrend", f"{bearish_pct}%", f"{downtrend} stocks")
        b3.metric("⚠️ Overbought (RSI≥70)", overbought)
        b4.metric("💧 Oversold (RSI≤30)", oversold)

        # Advance/decline style bar
        st.markdown("##### 📈 Trend Distribution")
        trend_counts = scan_df["Trend"].value_counts()
        tfig = go.Figure(go.Bar(
            x=trend_counts.values, y=trend_counts.index, orientation="h",
            marker_color=["#10b981" if "Up" in str(t) else "#ef4444" if "Down" in str(t)
                          else "#8e8e93" for t in trend_counts.index]))
        tfig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10),
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font=dict(color=theme_t.get("text", "#fff")))
        tfig.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
        st.plotly_chart(tfig, use_container_width=True)

        # Sector breadth
        st.markdown("##### 🏭 Sector Breadth (% of sector in uptrend)")
        sec_breadth = []
        for sec in scan_df["Sector"].unique():
            sub = scan_df[scan_df["Sector"] == sec]
            up = len(sub[sub["Trend"].isin(["Uptrend", "Strong Uptrend"])])
            pct = round(up / len(sub) * 100, 0) if len(sub) else 0
            sec_breadth.append({"Sector": sec, "Stocks": len(sub),
                                "In Uptrend": up, "Bullish %": pct})
        sb_df = pd.DataFrame(sec_breadth).sort_values("Bullish %", ascending=False)
        st.dataframe(sb_df, width="stretch", hide_index=True)
        st.caption("💡 Sectors at the top have the strongest internal breadth — "
                   "where the buying is concentrated right now.")

# ── Custom Screener ────────────────────────────────────────────────────────────
elif _page == 'screener':
    st.markdown('<div class="sec">🔬 Custom Screener</div>', unsafe_allow_html=True)
    st.caption("Filter the universe by your own criteria. Runs on the latest universe "
               "scan data (no extra fetch needed).")

    scan_df = st.session_state.get("scanner_cache")
    if scan_df is None or (hasattr(scan_df, "empty") and scan_df.empty):
        st.info("Screener needs universe scan data first.")
        if st.button("🔍 Run Universe Scan"):
            with st.spinner("Scanning universe…"):
                st.session_state.scanner_cache = generate_market_scanner()
            st.rerun()
    else:
        st.markdown(f"##### Filters ({len(scan_df):,} stocks in universe)")
        f1, f2, f3 = st.columns(3)
        with f1:
            rsi_range = st.slider("RSI range", 0, 100, (40, 70), key="scr_rsi")
            min_score = st.slider("Min signal score", -10, 20, 2, key="scr_score")
        with f2:
            trend_filter = st.multiselect(
                "Trend", ["Strong Uptrend", "Uptrend", "Sideways",
                          "Downtrend", "Strong Downtrend", "Recovery"],
                default=["Strong Uptrend", "Uptrend"], key="scr_trend")
            min_turnover = st.number_input("Min $ turnover (M)", 0.0, 5000.0, 5.0,
                                           key="scr_turnover")
        with f3:
            sectors_avail = sorted(scan_df["Sector"].unique().tolist())
            sector_filter = st.multiselect("Sectors (blank = all)", sectors_avail,
                                           default=[], key="scr_sector")
            signal_filter = st.multiselect(
                "Signal", ["🔥 STRONG BUY", "🟢 BUY SETUP", "🟡 ACCUMULATE",
                           "⚪ NEUTRAL", "🔴 AVOID"],
                default=[], key="scr_signal")

        # Apply filters
        res = scan_df.copy()
        res = res[(res["RSI"] >= rsi_range[0]) & (res["RSI"] <= rsi_range[1])]
        res = res[res["Score"] >= min_score]
        if trend_filter:
            res = res[res["Trend"].isin(trend_filter)]
        if "Turnover_M" in res.columns:
            res = res[res["Turnover_M"] >= min_turnover]
        if sector_filter:
            res = res[res["Sector"].isin(sector_filter)]
        if signal_filter:
            res = res[res["Signal"].isin(signal_filter)]

        st.markdown(f"##### 🎯 {len(res)} matches")
        if res.empty:
            st.warning("No stocks match these filters. Try loosening them.")
        else:
            show_cols = ["Stock", "Sector", "Signal", "Score", "CMP", "RSI",
                         "Trend", "VCP", "Trap", "RS", "Entry", "Target", "SL", "Turnover_M"]
            show_cols = [c for c in show_cols if c in res.columns]
            res_show = res[show_cols].sort_values("Score", ascending=False)
            _h = min(max(len(res_show) * 36 + 40, 200), 600)
            st.dataframe(res_show, width="stretch", height=_h, hide_index=True,
                         column_config={"Stock": st.column_config.TextColumn(pinned=True)})
            st.download_button(
                "⬇️ Export Screener Results CSV",
                res_show.to_csv(index=False).encode("utf-8"),
                file_name=f"screener_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv")

# ── Earnings Calendar ──────────────────────────────────────────────────────────
elif _page == 'earnings':
    st.markdown('<div class="sec">📆 Earnings Calendar</div>', unsafe_allow_html=True)
    st.caption("Upcoming results & key dates for your holdings. Helps you avoid being "
               "caught holding through a risky earnings event.")

    if df.empty:
        st.info("Add holdings to see their upcoming earnings dates.")
    else:
        hold_syms = sorted(df[df["status"] == "Open"]["stock"].unique().tolist())
        if not hold_syms:
            st.info("No open positions. Earnings calendar tracks your active holdings.")
        else:
            if st.button("📅 Fetch Earnings Dates", width="stretch"):
                import yfinance as _yf
                rows = []
                prog = st.progress(0.0)
                for i, sym in enumerate(hold_syms):
                    try:
                        t = _yf.Ticker(sym)
                        cal = None
                        try:
                            cal = t.calendar
                        except Exception:
                            cal = None
                        edate = None
                        if isinstance(cal, dict):
                            ev = cal.get("Earnings Date")
                            if ev:
                                edate = ev[0] if isinstance(ev, list) else ev
                        elif cal is not None and hasattr(cal, "loc"):
                            try:
                                edate = cal.loc["Earnings Date"][0]
                            except Exception:
                                edate = None
                        rows.append({"Stock": sym,
                                     "Next Earnings": str(edate)[:10] if edate else "—"})
                    except Exception:
                        rows.append({"Stock": sym, "Next Earnings": "—"})
                    prog.progress((i + 1) / len(hold_syms))
                prog.empty()
                st.session_state._earnings_cache = pd.DataFrame(rows)

            ecache = st.session_state.get("_earnings_cache")
            if ecache is not None and not ecache.empty:
                st.dataframe(ecache, width="stretch", hide_index=True)
                st.caption("⚠️ Earnings dates from Yahoo can be estimates and aren't "
                           "always available for every NSE stock. Verify with the "
                           "company / exchange before trading around results.")
            else:
                st.info("Click **Fetch Earnings Dates** to load upcoming results for "
                        "your holdings.")

# ── IPO Tracker ────────────────────────────────────────────────────────────────
elif _page == 'ipo':
    st.markdown('<div class="sec">🆕 IPO Tracker</div>', unsafe_allow_html=True)
    st.caption("Recently listed stocks and their performance since listing.")

    st.info("📌 **Honest note:** Reliable free IPO/GMP data is hard to get "
            "programmatically (NSE blocks cloud IPs; GMP sources are unofficial). "
            "This tracker lets you manually add stocks you're watching and tracks "
            "their performance via Yahoo once they're listed.")

    # Manual IPO watchlist using a session list
    if "_ipo_watch" not in st.session_state:
        st.session_state._ipo_watch = []

    ic1, ic2 = st.columns([3, 1])
    with ic1:
        _ipo_sym = st.text_input("Add newly listed stock (NSE symbol)", key="ipo_sym",
                                 placeholder="e.g. VEDPOWER")
    with ic2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("➕ Track", width="stretch"):
            if _ipo_sym and _ipo_sym.upper() not in st.session_state._ipo_watch:
                st.session_state._ipo_watch.append(_ipo_sym.upper().strip())
                st.rerun()

    if st.session_state._ipo_watch:
        if st.button("🔄 Refresh IPO Performance"):
            st.session_state._ipo_refresh = True

        rows = []
        for sym in st.session_state._ipo_watch:
            try:
                hist = fetch_chart_data(sym, "1mo", "1d")
                if hist is not None and not hist.empty:
                    listing_price = float(hist["Open"].iloc[0])
                    cur = float(hist["Close"].iloc[-1])
                    # live CMP if available
                    try:
                        lp = _cached_prices((sym,)).get(sym)
                        if lp: cur = lp
                    except Exception:
                        pass
                    chg = (cur / listing_price - 1) * 100 if listing_price else 0
                    rows.append({"Stock": sym, "Listing ~Price": round(listing_price, 2),
                                 "CMP": round(cur, 2), "Since Listing %": round(chg, 1),
                                 "Days": len(hist)})
                else:
                    rows.append({"Stock": sym, "Listing ~Price": "—", "CMP": "—",
                                 "Since Listing %": "—", "Days": 0})
            except Exception:
                rows.append({"Stock": sym, "Listing ~Price": "—", "CMP": "—",
                             "Since Listing %": "—", "Days": 0})

        ipo_df = pd.DataFrame(rows)
        st.dataframe(ipo_df, width="stretch", hide_index=True)

        # Remove option
        _rm = st.selectbox("Remove from tracker", ["—"] + st.session_state._ipo_watch,
                           key="ipo_rm")
        if _rm != "—" and st.button("🗑 Remove"):
            st.session_state._ipo_watch.remove(_rm)
            st.rerun()
        st.caption("💡 'Listing ~Price' is the first available open from Yahoo, which "
                   "approximates the listing-day open. Newly listed stocks need ~20 "
                   "trading days before full technical signals work (see Active Signals).")
    else:
        st.info("Add a recently listed stock above to start tracking its performance.")

# ── VCP Scanner ────────────────────────────────────────────────────────────────
elif _page == 'vcp':
    st.markdown('<div class="sec">📐 VCP Scanner — Volatility Contraction Pattern</div>',
                unsafe_allow_html=True)
    st.caption("Minervini's VCP: a leader basing through progressively tighter pullbacks "
               "with volume drying up, coiled under a pivot. Ready bases are near breakout.")

    if scan_for_vcp is None:
        st.warning("⚠️ VCP scanner not available — make sure the latest `signals.py` is deployed.")
    else:
        vc1, vc2, vc3 = st.columns([1, 1, 1])
        with vc1:
            _vcp_quality = st.selectbox("Min base quality", ["C", "B", "A", "A+"],
                                        index=1, key="vcp_q")
        with vc2:
            _vcp_ready = st.checkbox("Pivot-ready only", value=False, key="vcp_ready_only",
                                     help="Only bases coiled right under the breakout pivot")
        with vc3:
            st.markdown("<br>", unsafe_allow_html=True)
            _run_vcp = st.button("🔍 Scan for VCP", width="stretch")

        if _run_vcp:
            with st.spinner("Scanning universe for VCP bases… (this can take a minute)"):
                st.session_state.vcp_scan_cache = scan_for_vcp(
                    min_quality=_vcp_quality, ready_only=_vcp_ready)

        vres = st.session_state.get("vcp_scan_cache")
        if vres is None:
            st.info("Click **Scan for VCP** to find contraction bases across the universe.")
        elif not vres.get("vcp_setups"):
            st.warning(f"No VCP bases found at {_vcp_quality}+ quality. "
                       f"Scanned {vres.get('liquid',0)} liquid stocks. "
                       "Try lowering the quality filter or unchecking pivot-ready.")
        else:
            setups = vres["vcp_setups"]
            st.markdown(f"##### 🎯 {len(setups)} VCP bases "
                        f"({vres.get('ready_count',0)} pivot-ready) · "
                        f"scanned {vres.get('liquid',0)} liquid stocks")

            # ── Top 5 candidates card (already ranked: ready first, then quality) ──
            top5 = setups[:5]
            if top5:
                cards = ""
                for i, s in enumerate(top5, 1):
                    rdy = "⚡" if s["vcp_ready"] else ""
                    cards += (
                        f'<div style="display:flex;justify-content:space-between;'
                        f'padding:.5rem .7rem;border-bottom:1px solid var(--border)">'
                        f'<span><b style="color:var(--accent)">{i}.</b> '
                        f'<b>{s["stock"]}</b> {rdy} '
                        f'<span style="color:var(--muted);font-size:.78rem">{s["sector"]}</span></span>'
                        f'<span style="font-family:monospace;font-size:.82rem">'
                        f'VCP {s["quality"]} · pivot ${s["pivot"]} '
                        f'({s.get("pivot_distance_pct",0):+.1f}%)</span></div>')
                st.markdown(
                    f'<div style="background:var(--gradient);border:1px solid var(--accent);'
                    f'border-radius:12px;padding:1rem 1.2rem;margin-bottom:1.2rem">'
                    f'<div style="font-size:.8rem;font-weight:800;color:var(--accent);'
                    f'text-transform:uppercase;letter-spacing:.06em;margin-bottom:.5rem">'
                    f'⭐ Top 5 Candidates to Research</div>{cards}'
                    f'<div style="font-size:.72rem;color:var(--muted);margin-top:.6rem">'
                    f'Ranked: pivot-ready first, then base quality. These are research '
                    f'candidates — confirm the breakout above pivot before entering.</div>'
                    f'</div>', unsafe_allow_html=True)

            for s in setups:
                q = s["quality"]
                q_clr = {"A+": "#10b981", "A": "#22c55e",
                         "B": "#f59e0b", "C": "#8e8e93"}.get(q, "#8e8e93")
                ready_badge = ('<span style="background:rgba(16,185,129,.2);color:#10b981;'
                               'padding:.15rem .5rem;border-radius:5px;font-size:.7rem;'
                               'font-weight:800;margin-left:.5rem">⚡ PIVOT-READY</span>'
                               if s["vcp_ready"] else '')
                contractions_str = " → ".join(f"-{x}%" for x in s.get("contractions", []))
                rr_str = f"1:{s['risk_reward']}" if s.get("risk_reward") else "—"
                st.markdown(f"""
<div style="background:var(--card);border:1px solid var(--border);
            border-left:3px solid {q_clr};border-radius:10px;
            padding:1rem 1.2rem;margin-bottom:.7rem">
  <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:.4rem">
    <div>
      <b style="font-size:1.02rem">{s['stock']}</b>
      <span style="color:var(--muted);font-size:.78rem;margin-left:.4rem">{s['sector']}</span>
      <span style="background:{q_clr}22;color:{q_clr};padding:.12rem .5rem;border-radius:5px;
            font-size:.72rem;font-weight:800;margin-left:.5rem">VCP {q}</span>{ready_badge}
    </div>
    <div style="font-family:'JetBrains Mono',monospace;font-size:.82rem;color:var(--muted)">
      CMP ${s['cmp']} · Pivot ${s['pivot']}
    </div>
  </div>
  <div style="font-size:.82rem;color:var(--muted);margin-top:.5rem">
    📉 Contractions: <b style="color:var(--text)">{contractions_str}</b>
    &nbsp;·&nbsp; {s.get('detail','')}
  </div>
  <div style="font-size:.85rem;margin-top:.5rem">
    <b>Entry</b> ${s.get('entry','—')} &nbsp;·&nbsp;
    <b>Target</b> ${s.get('target','—')} &nbsp;·&nbsp;
    <b>Stop</b> ${s.get('stop_loss','—')} &nbsp;·&nbsp;
    <b>R:R</b> {rr_str}
  </div>
</div>""", unsafe_allow_html=True)

            # CSV export
            import pandas as _pd
            exp_df = _pd.DataFrame([{
                "Stock": s["stock"], "Sector": s["sector"], "Quality": s["quality"],
                "Ready": s["vcp_ready"], "CMP": s["cmp"], "Pivot": s["pivot"],
                "Pivot Dist %": s.get("pivot_distance_pct"),
                "Contractions": " → ".join(f"-{x}%" for x in s.get("contractions", [])),
                "Entry": s.get("entry"), "Target": s.get("target"),
                "Stop": s.get("stop_loss"), "R:R": s.get("risk_reward"),
            } for s in setups])
            st.download_button(
                "⬇️ Export VCP Results CSV",
                exp_df.to_csv(index=False).encode("utf-8"),
                file_name=f"vcp_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv")
            st.caption("⚠️ VCP works best on liquid leaders in an uptrend. A base forming "
                       "is not a buy by itself — the classic entry is a breakout ABOVE the "
                       "pivot on a volume surge. Always confirm before entering.")

# ── Relative Strength Leaders ──────────────────────────────────────────────────
elif _page == 'rs':
    st.markdown('<div class="sec">💪 Relative Strength Leaders</div>',
                unsafe_allow_html=True)
    st.caption("How each stock performs versus the S&P 500. RS Rating is a 1-99 percentile "
               "(IBD-style): 80+ = market leader, under 30 = laggard. Leaders tend to "
               "keep leading — the best longs are high-RS stocks, not cheap laggards.")

    if scan_relative_strength is None:
        st.warning("⚠️ RS scanner not available — make sure the latest `signals.py` is deployed.")
    else:
        rc1, rc2, rc3 = st.columns([1, 1, 1])
        with rc1:
            _rs_min = st.slider("Min RS Rating", 0, 99, 70, key="rs_min",
                                help="80+ = strong leaders. Filter out laggards.")
        with rc2:
            _rs_top = st.selectbox("Show top", ["All", "20", "50", "100"], index=2, key="rs_top")
        with rc3:
            st.markdown("<br>", unsafe_allow_html=True)
            _run_rs = st.button("🔍 Rank Universe by RS", width="stretch")

        if _run_rs:
            top_n = None if _rs_top == "All" else int(_rs_top)
            with st.spinner("Ranking the universe by relative strength… (takes a minute)"):
                st.session_state.rs_scan_cache = scan_relative_strength(
                    top_n=top_n, min_rating=_rs_min)

        rres = st.session_state.get("rs_scan_cache")
        if rres is None:
            st.info("Click **Rank Universe by RS** to find the market leaders.")
        elif rres.get("error"):
            st.error(f"⚠️ {rres['error']}. S&P 500 data may be temporarily unavailable — try again.")
        elif not rres.get("leaders"):
            st.warning(f"No stocks at RS Rating ≥ {_rs_min}. Lower the filter to see more.")
        else:
            leaders = rres["leaders"]
            nifty = rres.get("nifty_returns", {})
            # S&P 500 benchmark context
            st.markdown(
                f'<div style="background:var(--card);border:1px solid var(--border);'
                f'border-radius:10px;padding:.8rem 1.1rem;margin-bottom:1rem;'
                f'font-family:\'JetBrains Mono\',monospace;font-size:.82rem;color:var(--muted)">'
                f'📊 S&P 500 benchmark — 1M: <b style="color:var(--text)">{nifty.get("21",0):+.1f}%</b> · '
                f'3M: <b style="color:var(--text)">{nifty.get("63",0):+.1f}%</b> · '
                f'6M: <b style="color:var(--text)">{nifty.get("126",0):+.1f}%</b> · '
                f'1Y: <b style="color:var(--text)">{nifty.get("252",0):+.1f}%</b></div>',
                unsafe_allow_html=True)

            st.markdown(f"##### 🏆 {len(leaders)} leaders (RS ≥ {_rs_min}) · "
                        f"scanned {rres.get('liquid',0)} liquid stocks")

            # ── Top 5 leaders card (already sorted by RS rating desc) ──────────
            top5 = leaders[:5]
            if top5:
                cards = ""
                for i, l in enumerate(top5, 1):
                    vcp_mark = "🎯" if l.get("vcp") else ""
                    cards += (
                        f'<div style="display:flex;justify-content:space-between;'
                        f'padding:.5rem .7rem;border-bottom:1px solid var(--border)">'
                        f'<span><b style="color:var(--accent)">{i}.</b> '
                        f'<b>{l["stock"]}</b> {vcp_mark} '
                        f'<span style="color:var(--muted);font-size:.78rem">{l["sector"]}</span></span>'
                        f'<span style="font-family:monospace;font-size:.82rem">'
                        f'RS {l["rs_rating"]}/99 · 3M {l.get("ret_63d",0):+.0f}%</span></div>')
                st.markdown(
                    f'<div style="background:var(--gradient);border:1px solid var(--accent);'
                    f'border-radius:12px;padding:1rem 1.2rem;margin-bottom:1.2rem">'
                    f'<div style="font-size:.8rem;font-weight:800;color:var(--accent);'
                    f'text-transform:uppercase;letter-spacing:.06em;margin-bottom:.5rem">'
                    f'⭐ Top 5 Strongest Leaders</div>{cards}'
                    f'<div style="font-size:.72rem;color:var(--muted);margin-top:.6rem">'
                    f'Ranked by RS rating. 🎯 = also forming a VCP base. Research '
                    f'candidates, not buy signals — confirm your entry setup first.</div>'
                    f'</div>', unsafe_allow_html=True)

            # ── A-TIER OVERLAP: leaders that ALSO have a VCP base (best setups) ──
            overlap = [l for l in leaders if l.get("vcp")]
            if overlap:
                ov_cards = ""
                for l in overlap[:8]:
                    rdy = "⚡READY" if l.get("vcp_ready") else "base"
                    ov_cards += (
                        f'<div style="display:flex;justify-content:space-between;'
                        f'padding:.45rem .7rem;border-bottom:1px solid rgba(16,185,129,.2)">'
                        f'<span><b>{l["stock"]}</b> '
                        f'<span style="color:var(--muted);font-size:.76rem">{l["sector"]}</span></span>'
                        f'<span style="font-family:monospace;font-size:.8rem;color:#10b981">'
                        f'RS {l["rs_rating"]} · VCP {rdy}</span></div>')
                st.markdown(
                    f'<div style="background:rgba(16,185,129,.08);border:1px solid #10b981;'
                    f'border-radius:12px;padding:1rem 1.2rem;margin-bottom:1.2rem">'
                    f'<div style="font-size:.8rem;font-weight:800;color:#10b981;'
                    f'text-transform:uppercase;letter-spacing:.06em;margin-bottom:.5rem">'
                    f'💎 A-Tier Setups — Leader + VCP Base ({len(overlap)})</div>{ov_cards}'
                    f'<div style="font-size:.72rem;color:var(--muted);margin-top:.6rem">'
                    f'These appear on BOTH lists: strong RS leadership AND a low-risk VCP '
                    f'base. This overlap is the classic Minervini setup — your highest-'
                    f'conviction shortlist. Still confirm the breakout before entering.</div>'
                    f'</div>', unsafe_allow_html=True)

            # Build a clean table
            import pandas as _pd
            tbl = _pd.DataFrame([{
                "Stock": l["stock"],
                "RS Rating": l["rs_rating"],
                "RS Ratio": l["rs_ratio"],
                "Sector": l["sector"],
                "CMP": l["cmp"],
                "1M %": l.get("ret_21d"),
                "3M %": l.get("ret_63d"),
                "1Y %": l.get("ret_252d"),
                "Trend": l.get("trend"),
                "VCP": "🎯" if l.get("vcp") else "",
            } for l in leaders])
            _h = min(max(len(tbl) * 36 + 40, 200), 640)
            st.dataframe(tbl, width="stretch", height=_h, hide_index=True,
                         column_config={
                             "Stock": st.column_config.TextColumn(pinned=True),
                             "RS Rating": st.column_config.ProgressColumn(
                                 "RS Rating", min_value=0, max_value=99, format="%d"),
                         })
            st.download_button(
                "⬇️ Export RS Rankings CSV",
                tbl.to_csv(index=False).encode("utf-8"),
                file_name=f"rs_leaders_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv")
            st.caption("💡 The sweet spot: a high-RS leader (80+) forming a VCP base (🎯) "
                       "near its pivot. That's the classic Minervini long setup — strength "
                       "plus a low-risk entry point.")

# ── Post-render background deep scan ───────────────────────────────────────────
# The ENTIRE page has now rendered — you can see and interact with everything.
# Only NOW do we run one deep-scan stage (silently, in the background), then
# trigger a gentle rerun to advance to the next stage. Because this happens AFTER
# render, your current page/scroll/inputs are never interrupted mid-view.
if st.session_state.get("_run_deep_now", False):
    st.session_state._run_deep_now = False
    _stage = st.session_state.get("_deep_stage", "sector")
    st.session_state._deep_progress = _stage

    if _stage == "sector":
        try:
            st.session_state.sector_cache = sector_rotation()
            if (st.session_state.sector_cache is not None and
                    not st.session_state.sector_cache.empty):
                st.session_state.outlook_cache = predict_sector_outlook(
                    st.session_state.sector_cache)
                st.session_state.picks_cache = find_sector_picks(
                    st.session_state.sector_cache.head(5)["sector"].tolist(), 3)
            else:
                st.session_state.outlook_cache = pd.DataFrame()
                st.session_state.picks_cache   = []
        except Exception:
            pass
        st.session_state._deep_stage = "universe"

    elif _stage == "universe":
        try:
            _usd = generate_market_scanner()
            st.session_state.scanner_cache = (
                _usd if (_usd is not None and not _usd.empty) else pd.DataFrame())
        except Exception:
            pass
        st.session_state._deep_stage = "smc"

    elif _stage == "smc":
        try:
            if scan_for_smc_setups is not None:
                st.session_state.smc_scan_cache = scan_for_smc_setups(
                    min_quality="B", action_filter="All")
        except Exception:
            pass
        st.session_state._deep_stage = "traps"

    elif _stage == "traps":
        try:
            if scan_for_traps is not None:
                st.session_state.trap_scan_cache = scan_for_traps(min_confidence=55)
        except Exception:
            pass
        st.session_state._deep_stage = "vcp"

    elif _stage == "vcp":
        try:
            if scan_for_vcp is not None:
                st.session_state.vcp_scan_cache = scan_for_vcp(
                    min_quality="B", ready_only=False)
        except Exception:
            pass
        st.session_state._deep_stage = "rs"

    elif _stage == "rs":
        try:
            if scan_relative_strength is not None:
                st.session_state.rs_scan_cache = scan_relative_strength(
                    top_n=None, min_rating=0)
        except Exception:
            pass
        st.session_state._deep_stage = "sector"        # reset for next cycle
        st.session_state._deep_running = False          # whole sequence complete
        st.session_state._deep_progress = "done"
        st.session_state.last_slow_scan = time.time()   # mark deep scan complete

    # Advance to the next stage on the next pass. If the sequence is still
    # running, gently rerun; otherwise stop (no more forced reruns).
    if st.session_state.get("_deep_running", False):
        time.sleep(0.05)
        st.rerun()

# ── First-paint kickoff (login) ────────────────────────────────────────────────
# On first login, trigger one rerun so the deferred fast/deep scans begin.
if st.session_state.get("_kickoff_scan", False):
    st.session_state._kickoff_scan = False
    time.sleep(0.1)
    st.rerun()
