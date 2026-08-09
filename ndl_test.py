# ndl_test.py
import os, io, requests, pandas as pd
import nasdaqdatalink as ndl

API_KEY = os.getenv("NDL_API_KEY")  # leave unset for Test 1
API_KEY = "yfYQgyzaNzNGRWyVYzWH"

# Make the SDK look like a browser (helps sometimes)
try:
    from nasdaqdatalink.connection import Connection
    Connection.session.headers.update({
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
        "Accept": "application/json",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
    })
except Exception:
    pass

def get_csv_with_fallback(code: str, params: dict, label: str) -> pd.DataFrame:
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
        "Accept": "text/csv,application/json;q=0.9,*/*;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
    }
    last_err = None
    for base in ("https://data.nasdaq.com", "https://www.quandl.com"):
        url = f"{base}/api/v3/datasets/{code}.csv"
        try:
            r = requests.get(url, params=params, headers=headers, timeout=30)
            ct = (r.headers.get("Content-Type") or "").lower()
            print(f"[{label}] {base} status={r.status_code} content-type={ct}")
            # If 2xx, parse
            if 200 <= r.status_code < 300:
                df = pd.read_csv(io.StringIO(r.text))
                if "Date" in df.columns:
                    df["Date"] = pd.to_datetime(df["Date"])
                    df = df.set_index("Date").sort_index()
                return df
            # If 4xx/5xx on first domain, try next domain BEFORE raising
            last_err = requests.HTTPError(f"{r.status_code} for {url}")
            continue
        except Exception as e:
            last_err = e
            continue
    # ran out of fallbacks
    raise last_err if last_err else RuntimeError("Unknown error (no response)")

print("\n=== TEST 1: FRED/GDP (FREE) — CSV, NO API KEY ===")
try:
    # NOTE: do NOT include api_key here for the free test
    df = get_csv_with_fallback("FRED/GDP", {"rows": 5, "start_date": "2024-01-01"}, "FRED/GDP CSV no-key")
    print(df.head())
    print("[OK] Network path allows CSV from at least one domain.")
except Exception as e:
    print(f"[FAIL] FRED no-key CSV failed: {e}")
    print("This is a network/WAF block. Try from a home network or ask IT to whitelist data.nasdaq.com & www.quandl.com.")
    raise

print("\n=== TEST 2: CHRIS/ICE_TFM1 (TTF front-month) — CSV ===")
try:
    params = {"start_date": "2024-01-01"}
    if API_KEY:
        params["api_key"] = API_KEY
    df_ttf = get_csv_with_fallback("CHRIS/ICE_TFM1", params, "CHRIS/ICE_TFM1 CSV")
    print(df_ttf.head())
    print("[OK] CHRIS CSV worked.")
except Exception as e:
    print(f"[TTF CSV ERROR] {e}")
    print("If Test 1 succeeded but this failed with a JSON body about permissions, it’s an account/dataset access issue;")
    print("otherwise it’s still WAF/IP reputation. Try a different network.")
