#!/usr/bin/env python3
"""
build_full_universe.py — Rebuild the COMPLETE US universe (NASDAQ + NYSE + AMEX).
================================================================================
Downloads the full daily-updated US-listed symbol set (symbol, name, sector,
market cap) and writes a comprehensive, market-cap-ordered us_universe.csv —
~6,800 stocks plus your curated us_etfs.csv.

Run whenever you want to refresh listings (new IPOs, delistings):
    pip install pandas requests
    python build_full_universe.py

Source: rreichel3/US-Stock-Symbols (daily mirror of the Nasdaq screener API).
Preferred-share / warrant lines (symbols containing '^') are dropped because
they don't resolve cleanly on Yahoo Finance.

NOTE: this gives the full EXCHANGE-LISTED set. Truly off-exchange tickers (OTC,
brand-new, or delisted names like some you add by hand) won't appear here — put
those in us_custom.csv, which is always loaded and always scanned.
================================================================================
"""
import io
import json
import os
import sys

import pandas as pd

try:
    import requests
except ImportError:
    sys.exit("Install requests:  pip install requests")

_BASE = os.path.dirname(os.path.abspath(__file__))
_RAW = "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main"
_EXCHANGES = ("nasdaq", "nyse", "amex")


def main():
    rows = []
    for ex in _EXCHANGES:
        url = f"{_RAW}/{ex}/{ex}_full_tickers.json"
        print(f"Downloading {ex} …")
        try:
            data = requests.get(url, timeout=30).json()
        except Exception as e:
            print(f"  failed: {e}")
            continue
        for r in data:
            sym = (r.get("symbol") or "").strip().upper()
            if not sym or "^" in sym:
                continue
            name = (r.get("name") or "").strip().replace(",", " ")
            sector = (r.get("sector") or "").strip() or "Others"
            mc = (r.get("marketCap") or "0").replace(",", "").strip()
            try:
                mc = float(mc)
            except Exception:
                mc = 0.0
            rows.append({"Symbol": sym, "Name": name, "Sector": sector, "_mc": mc})

    if not rows:
        sys.exit("No symbols downloaded — check your network connection.")

    stocks = (pd.DataFrame(rows)
              .drop_duplicates(subset=["Symbol"], keep="first")
              .sort_values("_mc", ascending=False)[["Symbol", "Name", "Sector"]])
    stocks.to_csv(os.path.join(_BASE, "us_stocks.csv"), index=False)

    # Merge curated ETFs (clean categories) after the market-cap-ordered stocks.
    etf_fp = os.path.join(_BASE, "us_etfs.csv")
    if os.path.exists(etf_fp):
        etfs = pd.read_csv(etf_fp)
        etfs.columns = [c.strip() for c in etfs.columns]
        etfs["Name"] = etfs["Name"].astype(str).str.replace(",", " ")
        combined = pd.concat([stocks, etfs[["Symbol", "Name", "Sector"]]], ignore_index=True)
    else:
        combined = stocks
    combined = combined.drop_duplicates(subset=["Symbol"], keep="first")
    combined.to_csv(os.path.join(_BASE, "us_universe.csv"), index=False)

    print(f"\n✅ us_universe.csv  →  {len(combined):,} symbols "
          f"({len(stocks):,} stocks + ETFs)")
    print("   Most-liquid first; the app scans the top N per the sidebar 'Scan depth'.")


if __name__ == "__main__":
    main()
