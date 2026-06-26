# 📈 US Swing Trading Dashboard

A dark-themed Streamlit dashboard for swing trading **US equities + ETFs** (NYSE / NASDAQ),
with live prices, signal generation, relative strength, market regime, screeners,
position tracking, watchlists, and alerts.

This is the **US standalone** sibling of the NSE dashboard — same engine, re-pointed
to US data (bare Yahoo tickers, S&P 500 benchmark, USD).

---

## 🌎 What's different from the NSE build

| Layer | NSE | US (this app) |
|-------|-----|---------------|
| Universe | Nifty / EQUITY_L CSVs | `us_universe.csv` (stocks + ETFs) |
| Ticker resolution | `SYMBOL.NS` / `.BO` | bare `SYMBOL` (share classes -> `BRK-B`) |
| RS benchmark | Nifty 50 `^NSEI` | S&P 500 `^GSPC` |
| Indices | Sensex, Nifty, Bank Nifty, India VIX | S&P 500, Nasdaq, Dow, Russell 2000, Nasdaq 100, VIX |
| Sector proxies | NSE sector indices | SPDR sector ETFs (XLK, XLF, XLV...) |
| Currency | INR | USD |
| Turnover units | Cr/day | $M/day |

---

## 🗂️ Universe (stocks + ETFs, liquidity-filtered)

- `us_stocks.csv` — ~4,720 US common stocks (Symbol, Name, Sector)
- `us_etfs.csv` — ~130 of the most liquid US ETFs (broad market, sector, bond, commodity, leveraged, thematic...)
- `us_universe.csv` — the two combined and de-duplicated (~4,847 symbols) — the app reads this by default.
- `us_universe_liquid.csv` — produced by `build_us_universe.py`; if present, the app prefers it.

### Refreshing the liquid universe
    pip install -r requirements.txt
    python build_us_universe.py

Default filter: price >= $5 AND 20-day average dollar volume >= $5M/day
(edit the constants at the top of build_us_universe.py to taste).

> Scanning ~4,800 tickers live on every page load would time out / rate-limit on
> Streamlit Cloud, so the liquid list is baked offline and the small CSV is shipped.

---

## 🚀 Run locally
    pip install -r requirements.txt
    streamlit run app.py        # opens http://localhost:8501

With no database configured, the app uses local SQLite (trades_us.db) automatically.

---

## ☁️ Deploy to Streamlit Cloud
1. Push this folder to a NEW GitHub repo (e.g. SwingDashboardUS).
2. On share.streamlit.io -> New app -> pick the repo -> main file app.py.
3. (Optional) Add Postgres in Settings -> Secrets — see .streamlit/secrets.toml.example.
   Use a SEPARATE database from the NSE app so US and NSE trades never mix.

---

## 🗄️ Database
- Default: local SQLite (trades_us.db) — zero setup, but resets on Streamlit Cloud redeploys.
- Persistent: add a [postgres] block to Streamlit secrets (Neon, Supabase, RDS...).
  The app auto-detects it and switches to Postgres. Do not reuse the NSE database.

> Unlike the original NSE repo, this build does NOT hardcode any DB credentials.
> Credentials come from Streamlit secrets / environment variables only.

---

## 📦 Dependencies
streamlit, pandas, numpy, yfinance, plotly, scipy, requests, psycopg2-binary,
streamlit-cookies-controller, streamlit-autorefresh (see requirements.txt).
