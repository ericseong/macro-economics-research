# eex_eua_smoke_test.py
# Quick check that we can fetch EUA Spot data from EEX and aggregate to daily EUR/t.

import argparse
import io
import sys
from datetime import datetime, timedelta

import pandas as pd
import requests

EEX_BASES = [
    "https://api1.datasource.eex-group.com",
    "https://api2.datasource.eex-group.com",  # fallback host
]

def _parse_decimal(x: str) -> float:
    """
    EEX often uses comma decimals (e.g., '74,12'). Convert robustly to float.
    Also strips thousands separators if present.
    """
    if pd.isna(x):
        return float("nan")
    s = str(x).strip()
    if not s:
        return float("nan")
    # Remove thousands separators and flip comma decimal to dot
    s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except Exception:
        return float("nan")

def fetch_eua_spot_trades_csv(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Download EEX EUA SPOT trade history (CSV) for the given date range.
    root=SEME (EEX EUA Spot), ProductType=SPOT.
    Returns the raw trade-level DataFrame.
    """
    last_err = None
    for base in EEX_BASES:
        url = f"{base}/getHistory/csv"
        params = {
            "root": "SEME",           # EEX EUA Spot
            "ProductType": "SPOT",
            "start": start_date,
            "end": end_date,
        }
        print(f"[INFO] Requesting: {url}  params={params}")
        try:
            r = requests.get(url, params=params, timeout=30)
            print(f"[INFO] HTTP {r.status_code}  content-type={r.headers.get('Content-Type')}")
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
            if df.empty:
                raise RuntimeError("CSV returned but table is empty.")
            df.columns = [c.strip() for c in df.columns]
            print(f"[INFO] Columns: {list(df.columns)}  rows={len(df)}")
            return df
        except Exception as e:
            print(f"[WARN] Fetch failed for host {base}: {e}")
            last_err = e
    raise SystemExit(f"[ERROR] All EEX hosts failed. Last error: {last_err}")

def daily_mean_from_trades(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a daily mean EUA price (EUR/t) from the raw trades CSV.
    Attempts to find a timestamp and a price column robustly.
    Returns columns: date, eua_eur_t
    """
    # Heuristics for time/price column names seen on EEX feeds
    cand_time = [c for c in df.columns if "time" in c.lower() or "date" in c.lower()]
    cand_price = [c for c in df.columns if "price" in c.lower() or "settle" in c.lower() or "close" in c.lower()]

    if not cand_time:
        raise RuntimeError("Couldn't find a timestamp column in EEX CSV.")
    if not cand_price:
        raise RuntimeError("Couldn't find a price column in EEX CSV.")

    time_col = cand_time[0]
    price_col = cand_price[0]

    # Convert
    ts = pd.to_datetime(df[time_col], errors="coerce")
    px = df[price_col].map(_parse_decimal)

    dd = pd.DataFrame({"date": ts.dt.normalize(), "price": px}).dropna()
    daily = (
        dd.groupby("date", as_index=False)["price"]
          .mean()
          .rename(columns={"price": "eua_eur_t"})
          .sort_values("date")
    )
    return daily

def main():
    ap = argparse.ArgumentParser(description="EEX EUA Spot smoke test")
    ap.add_argument("--start", type=str, default="", help="Start date YYYY-MM-DD (default: today-90d)")
    ap.add_argument("--end", type=str, default="", help="End date YYYY-MM-DD (default: today)")
    ap.add_argument("--days", type=int, default=90, help="If start not given, pull this many days back from today")
    args = ap.parse_args()

    end_dt = datetime.utcnow().date() if not args.end else datetime.strptime(args.end, "%Y-%m-%d").date()
    if args.start:
        start_dt = datetime.strptime(args.start, "%Y-%m-%d").date()
    else:
        start_dt = end_dt - timedelta(days=args.days)

    start_s = start_dt.strftime("%Y-%m-%d")
    end_s = end_dt.strftime("%Y-%m-%d")

    print(f"[INFO] Date range: {start_s} → {end_s}")

    # 1) Download raw trades CSV
    raw = fetch_eua_spot_trades_csv(start_s, end_s)

    # 2) Aggregate to daily mean EUR/t
    daily = daily_mean_from_trades(raw)

    # 3) Show preview and simple stats
    print("\n=== Daily EUA (EUR/t) preview ===")
    print(daily.head(10).to_string(index=False))
    print("\n=== Tail ===")
    print(daily.tail(10).to_string(index=False))
    print("\n=== Summary ===")
    print(daily["eua_eur_t"].describe().to_string())

    # 4) Optional: write a quick CSV (uncomment if you want a file)
    # daily.to_csv("eua_eex_daily.csv", index=False)
    # print("\n[INFO] Wrote eua_eex_daily.csv")

if __name__ == "__main__":
    try:
        main()
    except SystemExit as e:
        raise
    except Exception as e:
        print(f"[FATAL] {e}", file=sys.stderr)
        sys.exit(1)

