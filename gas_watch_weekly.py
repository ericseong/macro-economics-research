#!/usr/bin/env python3
"""
Gas Watch — Weekly Trend Forecaster (live storage, wind, FX, + real-data hooks)
===============================================================================

Purpose
-------
Forecast the *weekly* trend of European gas prices (TTF) using daily fundamentals and
market indicators. Ingests daily data (some live, some optional via public APIs or CSV),
engineers features, aggregates to weekly, fits a Ridge model (alpha tuned by TS-CV) for
returns or a Logistic classifier for direction, and outputs an explainable weekly view.

Live inputs wired now
---------------------
- EU storage fill % via AGSI+ (env: `AGSI_API_KEY` or file: `../credentials/credential_agsi_api.txt`).
- Wind proxy via Open-Meteo ERA5 daily wind.
- EUR/USD FX via ECB historical CSV (auto).
- Brent via EIA (env: `EIA_API_KEY` or file: `../credentials/credential_eia_api.txt`).
- HDD anomaly via Open-Meteo ERA5 daily temperature.

Optional market feeds (auto-hooked if provided)
----------------------------------------------
- TTF front-month (€/MWh) via Nasdaq Data Link (env: `NDL_API_KEY` + `NDL_TTF_CODE`
  or key file: `../credentials/credential_ndl_api.txt`) or CSV (`TTF_SOURCE_CSV`).
- EUA (€/tCO₂) via Nasdaq Data Link (`NDL_EUA_CODE`) or CSV (`EUA_SOURCE_CSV`).
- JKM ($/MMBtu) via Nasdaq Data Link (`NDL_JKM_CODE`) or CSV (`JKM_SOURCE_CSV`).

CSV format expectations (overridable): columns `date` and `value` by default.
You can override with `TTF_CSV_DATE_COL`, `TTF_CSV_VALUE_COL`, etc.

How to run
----------
Regression (pct change with guardrail + TS-CV):
    python gas_watch_weekly.py --run-all \
      --report-out ./output/weekly_gas_watch.md \
      --json-out   ./output/weekly_gas_watch.json

Classification (UP/DOWN + probs + TS-CV accuracy):
    python gas_watch_weekly.py --run-all --mode classify \
      --report-out ./output/weekly_gas_watch.md \
      --json-out   ./output/weekly_gas_watch.json

"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import io
import os
import time
import json

import numpy as np
import pandas as pd
import requests

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

# -----------------------------
# Configuration
# -----------------------------

@dataclass
class GasWatchConfig:
    direction_threshold: float = 0.01  # e.g., +/-1% defines Up/Down
    ridge_alpha: float = 1.0
    min_weeks_to_train: int = 60
    seed: int = 42

# -----------------------------
# Data Ingestion (AGSI+, Open-Meteo, ECB, EIA, NDL/CSV)
# -----------------------------

AGSI_BASE = "https://agsi.gie.eu/api"
OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
ECB_HIST_CSV = "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-hist.csv"
EIA_ENDPOINT = "https://api.eia.gov/series/"

def _to_date_str(ts: pd.Timestamp) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")

# ---- Credential helpers (env overrides, else file relative to this script) ----

def _read_key_file(relpath: str) -> str:
    try:
        base = Path(__file__).resolve().parent
    except NameError:
        base = Path(".").resolve()
    fpath = (base / relpath).resolve()
    if fpath.exists():
        try:
            return fpath.read_text(encoding="utf-8").strip()
        except Exception:
            return ""
    return ""

def get_agsi_api_key() -> str:
    return os.getenv("AGSI_API_KEY", "").strip() or _read_key_file("../quickndirty/credentials/credential_agsi_api.txt")

def get_eia_api_key() -> str:
    return os.getenv("EIA_API_KEY", "").strip() or _read_key_file("../quickndirty/credentials/credential_eia_api.txt")

def get_ndl_api_key() -> str:
    return os.getenv("NDL_API_KEY", "").strip() or _read_key_file("../quickndirty/credentials/credential_ndl_api.txt")

# ---- Utility loaders ----

def load_csv_series(path: str, date_col: str = "date", value_col: str = "value") -> pd.DataFrame:
    df = pd.read_csv(path)
    if date_col not in df.columns or value_col not in df.columns:
        raise ValueError("CSV missing required columns: {dc}, {vc}".format(dc=date_col, vc=value_col))
    out = df[[date_col, value_col]].copy()
    out[date_col] = pd.to_datetime(out[date_col])
    out = out.rename(columns={date_col: "date", value_col: "value"}).sort_values("date").dropna()
    return out

def fetch_ndl_timeseries(code: str, api_key: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Nasdaq Data Link generic fetcher. Returns columns: date, value."""
    url = "https://data.nasdaq.com/api/v3/datasets/{code}.json".format(code=code)
    params = {"api_key": api_key, "start_date": start_date, "end_date": end_date}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    j = r.json().get("dataset", {})
    data = j.get("data", [])
    colnames = j.get("column_names", [])
    if not data or not colnames:
        raise RuntimeError("NDL returned no data")
    date_idx = colnames.index("Date") if "Date" in colnames else 0
    val_idx = 1 if len(colnames) > 1 else 0
    rows = []
    for row in data:
        try:
            d = pd.to_datetime(row[date_idx])
            v = float(row[val_idx]) if row[val_idx] is not None else np.nan
            rows.append((d, v))
        except Exception:
            continue
    df = pd.DataFrame(rows, columns=["date", "value"]).dropna().sort_values("date")
    return df

# ---- Live fetchers ----

def fetch_yahoo_ttf(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch Dutch TTF front-month via Yahoo Finance (no API key).
    Returns columns: date, value (€/MWh), guaranteed 1-D.
    """
    try:
        import yfinance as yf  # pip install yfinance
    except ImportError as e:
        raise RuntimeError("yfinance not installed. Run: pip install yfinance") from e

    tickers_env = os.getenv("YF_TTF_TICKERS", "").strip()
    tickers = [t.strip() for t in tickers_env.split(",") if t.strip()] or ["TTF=F", "MTTF=F", "TFM=F"]

    # Yahoo end is exclusive; add a day
    end_plus = (pd.to_datetime(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    def _pick_close(df: pd.DataFrame) -> pd.Series | None:
        """
        Make sure we end up with a 1-D price series even if columns are MultiIndex.
        Prefer 'Adj Close', fallback to 'Close'.
        """
        if df is None or df.empty:
            return None

        # Single-level columns case
        for col in ("Adj Close", "Close"):
            if col in df.columns:
                s = df[col]
                # If this is a DataFrame (e.g., multiple tickers), pick the first non-empty column
                if isinstance(s, pd.DataFrame):
                    for c in s.columns:
                        if s[c].notna().any():
                            return s[c]
                    return None
                return s if s.notna().any() else None

        # MultiIndex columns case: try ('Adj Close', <ticker>) then ('Close', <ticker>)
        if isinstance(df.columns, pd.MultiIndex):
            lvl0 = df.columns.get_level_values(0)
            lvl1 = df.columns.get_level_values(1)
            for col0 in ("Adj Close", "Close"):
                if col0 in set(lvl0):
                    sub = df[col0]
                    # 'sub' may still be DataFrame with columns per ticker
                    if isinstance(sub, pd.DataFrame):
                        # choose first non-empty column
                        for c in sub.columns:
                            if sub[c].notna().any():
                                return sub[c]
        return None

    for tkr in tickers:
        try:
            df = yf.download(tkr, start=start_date, end=end_plus, interval="1d",
                             auto_adjust=False, progress=False, threads=False)
            s = _pick_close(df)
            if s is None:
                continue
            s = s.copy()
            # normalize and ensure 1-D float series
            s.index = pd.to_datetime(s.index).tz_localize(None)
            s = pd.to_numeric(s, errors="coerce").dropna()
            if s.empty:
                continue
            out = s.reset_index().rename(columns={"index": "date", "Date": "date", s.name: "value"})
            out["date"] = pd.to_datetime(out["date"])
            result = out[["date", "value"]].sort_values("date")
            #print(result)
            return result
        except Exception:
            continue

    raise RuntimeError("Yahoo Finance: no TTF ticker returned data. "
                       "Try setting YF_TTF_TICKERS or provide TTF_SOURCE_CSV/NDL.")


def fetch_yahoo_brent(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch Brent (USD/bbl) via Yahoo Finance (no API key).
    Tries a few tickers and returns columns: date, brent_usd_bbl.
    """
    try:
        import yfinance as yf  # pip install yfinance
    except ImportError as e:
        raise RuntimeError("yfinance not installed. Run: pip install yfinance") from e

    # Override with: export YF_BRENT_TICKERS="BZ=F,CO=F,BRENTOIL"
    tickers_env = os.getenv("YF_BRENT_TICKERS", "").strip()
    tickers = [t.strip() for t in tickers_env.split(",") if t.strip()] or ["BZ=F", "CO=F", "BRENTOIL"]

    end_plus = (pd.to_datetime(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    def _to_1d_price(df: pd.DataFrame):
        if df is None or df.empty:
            return None
        # Single-index
        for col in ("Adj Close", "Close"):
            if col in df.columns:
                s = df[col]
                if isinstance(s, pd.DataFrame):
                    # multiple tickers -> pick first non-empty
                    for c in s.columns:
                        sc = pd.to_numeric(s[c], errors="coerce").dropna()
                        if not sc.empty:
                            return sc
                    return None
                return pd.to_numeric(s, errors="coerce").dropna()
        # MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            for col0 in ("Adj Close", "Close"):
                if col0 in set(df.columns.get_level_values(0)):
                    sub = df[col0]
                    for c in sub.columns:
                        sc = pd.to_numeric(sub[c], errors="coerce").dropna()
                        if not sc.empty:
                            return sc
        return None

    for tkr in tickers:
        try:
            df = yf.download(tkr, start=start_date, end=end_plus, interval="1d",
                             auto_adjust=False, progress=False, threads=False)
            s = _to_1d_price(df)
            if s is None or s.empty:
                continue
            s.index = pd.to_datetime(s.index).tz_localize(None)
            s = s.sort_index()
            out = s.reset_index().rename(columns={"index": "date", "Date": "date", s.name: "brent_usd_bbl"})
            out["date"] = pd.to_datetime(out["date"])
            result = out[["date", "brent_usd_bbl"]]
            #print(result)
            return result
        except Exception:
            continue
    raise RuntimeError("Yahoo Finance: no Brent ticker returned data. "
                       "Try setting YF_BRENT_TICKERS or use EIA key.")


def fetch_agsi_eu_storage(start_date: str, end_date: str, api_key: str, page_size: int = 300) -> pd.DataFrame:
    headers = {"x-key": api_key}
    params = {"type": "eu", "from": start_date, "to": end_date, "size": page_size, "page": 1}
    rows = []
    while True:
        r = requests.get(AGSI_BASE, headers=headers, params=params, timeout=30)
        r.raise_for_status()
        j = r.json()
        for d in j.get("data", []):
            rows.append({
                "date": pd.to_datetime(d.get("gasDayStart")),
                "storage_fill_pct": float(d.get("full")) if d.get("full") is not None else np.nan,
                "gasInStorage_TWh": float(d.get("gasInStorage")) if d.get("gasInStorage") is not None else np.nan,
                "workingGasVolume_TWh": float(d.get("workingGasVolume")) if d.get("workingGasVolume") is not None else np.nan,
            })
        last_page = j.get("last_page", params["page"])
        if params["page"] >= last_page:
            break
        params["page"] += 1
        time.sleep(0.3)
    if not rows:
        raise RuntimeError("AGSI+ returned no rows")
    df = pd.DataFrame(rows).dropna(subset=["date"]).sort_values("date")
    if df["storage_fill_pct"].isna().any():
        with np.errstate(divide="ignore", invalid="ignore"):
            pct = 100.0 * df["gasInStorage_TWh"] / df["workingGasVolume_TWh"]
        df["storage_fill_pct"] = df["storage_fill_pct"].fillna(pct)
    print('--- EU Storage ---')
    print(df)
    return df

def fetch_open_meteo_wind_index(coords: List[Tuple[float, float]], start_date: str, end_date: str, speed_unit: str = "ms") -> pd.DataFrame:
    frames = []
    for (lat, lon) in coords:
        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": "wind_speed_10m_mean",
            "start_date": start_date,
            "end_date": end_date,
            "timezone": "UTC",
            "wind_speed_unit": speed_unit,
        }
        r = requests.get(OPEN_METEO_ARCHIVE, params=params, timeout=30)
        r.raise_for_status()
        j = r.json().get("daily", {})
        times = j.get("time", [])
        speeds = j.get("wind_speed_10m_mean", [])
        if not times:
            continue
        df = pd.DataFrame({"date": pd.to_datetime(times), "ws_{:.2f}_{:.2f}".format(lat, lon): speeds})
        frames.append(df)
    if not frames:
        raise RuntimeError("Open-Meteo returned no wind data")
    wind = frames[0]
    for f in frames[1:]:
        wind = wind.merge(f, on="date", how="outer")
    cols = [c for c in wind.columns if c.startswith("ws_")]
    wind["wind_speed_10m_mean"] = wind[cols].mean(axis=1)
    wind = wind.sort_values("date").dropna(subset=["wind_speed_10m_mean"]).reset_index(drop=True)
    mu = wind["wind_speed_10m_mean"].mean()
    wind["wind_index"] = wind["wind_speed_10m_mean"] / mu
    return wind[["date", "wind_speed_10m_mean", "wind_index"]]

def fetch_open_meteo_hdd_anom(coords: List[Tuple[float, float]], start_date: str, end_date: str, base_c: float = 18.0) -> pd.DataFrame:
    """Build HDD anomaly from ERA5 mean temperature across coords."""
    frames = []
    for (lat, lon) in coords:
        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": "temperature_2m_mean",
            "start_date": start_date,
            "end_date": end_date,
            "timezone": "UTC",
        }
        r = requests.get(OPEN_METEO_ARCHIVE, params=params, timeout=30)
        r.raise_for_status()
        j = r.json().get("daily", {})
        t = j.get("temperature_2m_mean", [])
        times = j.get("time", [])
        if not times:
            continue
        frames.append(pd.DataFrame({"date": pd.to_datetime(times), "t2m_{:.2f}_{:.2f}".format(lat, lon): t}))
    if not frames:
        raise RuntimeError("Open-Meteo returned no temperature data")
    tmp = frames[0]
    for f in frames[1:]:
        tmp = tmp.merge(f, on="date", how="outer")
    tcols = [c for c in tmp.columns if c.startswith("t2m_")]
    tmp["t2m_mean"] = tmp[tcols].mean(axis=1)
    tmp = tmp.sort_values("date").dropna(subset=["t2m_mean"]).reset_index(drop=True)
    tmp["hdd"] = (base_c - tmp["t2m_mean"]).clip(lower=0)
    tmp["doy"] = tmp["date"].dt.dayofyear
    clim = tmp.groupby("doy")["hdd"].mean()
    tmp["hdd_anom"] = tmp.apply(lambda r: r["hdd"] - clim.loc[r["doy"]], axis=1)
    return tmp[["date", "hdd_anom"]]

def fetch_ecb_eur_usd(start_date: str, end_date: str) -> pd.DataFrame:
    r = requests.get(ECB_HIST_CSV, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    if "Date" not in df.columns or "USD" not in df.columns:
        raise RuntimeError("ECB CSV format unexpected")
    df = df.rename(columns={"Date": "date"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    df["eur_per_usd"] = 1.0 / pd.to_numeric(df["USD"], errors="coerce")
    df = df.dropna(subset=["eur_per_usd"])
    mask = (df["date"] >= pd.to_datetime(start_date)) & (df["date"] <= pd.to_datetime(end_date))
    return df.loc[mask, ["date", "eur_per_usd"]]

def fetch_eia_brent(start_date: str, end_date: str, api_key: str) -> pd.DataFrame:
    """EIA Europe Brent Spot Price (series PET.RBRTE.D), USD/bbl."""
    params = {"api_key": api_key, "series_id": "PET.RBRTE.D"}
    r = requests.get(EIA_ENDPOINT, params=params, timeout=30)
    r.raise_for_status()
    series = r.json().get("series", [])
    if not series:
        raise RuntimeError("EIA returned no series")
    data = series[0].get("data", [])  # [ [YYYYMMDD, value], ... ]
    rows = []
    for d, v in data:
        try:
            dt = pd.to_datetime(str(d))
            rows.append((dt, float(v)))
        except Exception:
            continue
    df = pd.DataFrame(rows, columns=["date", "brent_usd_bbl"]).sort_values("date")
    mask = (df["date"] >= pd.to_datetime(start_date)) & (df["date"] <= pd.to_datetime(end_date))
    return df.loc[mask]

# --- Loader orchestrator ---

def load_or_synthesize_daily_data(n_days: int = 3 * 365, seed: int = 42) -> pd.DataFrame:
    """Return a daily dataframe combining live overlays and optional market feeds.
    Columns (some may be overlaid by live sources):
      - ttf_eur_mwh (target proxy if no real target wired)
      - jkm_usd_mmbtu
      - storage_fill_pct, storage_dev_5y_pp
      - norway_outage (placeholder unless wired)
      - wind_index
      - hdd_anom
      - eua_eur_t
      - brent_usd_bbl
      - eur_per_usd (for spreads)
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=n_days, freq="D")

    # --- Synthetic base so pipeline always runs ---
    ttf_base = 45 + 10 * np.sin(np.linspace(0, 8 * np.pi, n_days))
    eua_base = 70 + 5 * np.sin(np.linspace(0, 4 * np.pi, n_days) + 0.8)
    jkm_base = 12 + 2 * np.sin(np.linspace(0, 6 * np.pi, n_days) + 1.0)

    ttf_noise = rng.normal(0, 1.2, n_days)
    eua_noise = rng.normal(0, 0.6, n_days)
    jkm_noise = rng.normal(0, 0.3, n_days)

    outage = np.zeros(n_days)
    for start in rng.integers(20, n_days - 30, size=10):
        outage[start : start + rng.integers(5, 12)] = 1

    wind_index = 1.0 + 0.2 * np.sin(np.linspace(0, 10 * np.pi, n_days)) + rng.normal(0, 0.1, n_days)
    hdd_anom = 0.0 + 2.0 * np.sin(np.linspace(0, 3 * np.pi, n_days) - 0.7) + rng.normal(0, 0.8, n_days)
    storage_fill = np.clip(30 + 0.07 * np.arange(n_days) % 100 + rng.normal(0, 2.0, n_days), 5, 100)
    storage_dev = rng.normal(0, 3.0, n_days)
    brent = 75 + 3 * np.sin(np.linspace(0, 3 * np.pi, n_days) + 2.1) + rng.normal(0, 0.8, n_days)

    ttf = (
        ttf_base
        + 2.2 * outage
        - 1.5 * (wind_index - wind_index.mean())
        - 0.03 * (storage_fill - storage_fill.mean())
        + 0.08 * (eua_base - eua_base.mean())
        + 0.18 * hdd_anom
        + ttf_noise
    )

    eua = eua_base + eua_noise
    jkm = jkm_base + jkm_noise

    df = pd.DataFrame({
        "date": dates,
        "ttf_eur_mwh": ttf,
        "jkm_usd_mmbtu": jkm,
        "storage_fill_pct": storage_fill,
        "storage_dev_5y_pp": storage_dev,
        "norway_outage": outage,
        "wind_index": wind_index,
        "hdd_anom": hdd_anom,
        "eua_eur_t": eua,
        "brent_usd_bbl": brent,
    }).set_index("date").sort_index()

    # --- Live overlays ---
    start_date = _to_date_str(df.index.min())
    end_date   = _to_date_str(df.index.max())

    # 1) EU storage (AGSI+)
    agsi_key = get_agsi_api_key()
    #print('agsi_key')
    #print(agsi_key)
    if agsi_key:
        try:
            agsi = fetch_agsi_eu_storage(start_date, end_date, agsi_key).set_index("date").sort_index()
            df["storage_fill_pct"] = df.index.to_series().map(agsi["storage_fill_pct"]).combine_first(df["storage_fill_pct"])  # prefer live
            tmp = agsi[["storage_fill_pct"]].dropna().copy()
            tmp["doy"] = tmp.index.dayofyear
            clim = tmp.groupby("doy")["storage_fill_pct"].mean()
            df["storage_dev_5y_pp"] = df["storage_fill_pct"] - df.index.dayofyear.map(clim)
        except Exception as e:
            print("[WARN] AGSI+ fetch failed: {}".format(e))

    # 2) Wind proxy (Open-Meteo)
    coords = [(52.37,4.90),(51.16,10.45),(56.26,9.50),(48.86,2.35),(40.42,-3.70),(54.0,-2.0)]
    try:
        wind = fetch_open_meteo_wind_index(coords, start_date, end_date).set_index("date").sort_index()
        df["wind_index"] = df.index.to_series().map(wind["wind_index"]).combine_first(df["wind_index"])  # prefer live
    except Exception as e:
        print("[WARN] Open-Meteo wind fetch failed: {}".format(e))

    # 3) HDD anomaly (Open-Meteo temps)
    try:
        hdd = fetch_open_meteo_hdd_anom(coords, start_date, end_date).set_index("date").sort_index()
        df["hdd_anom"] = df.index.to_series().map(hdd["hdd_anom"]).combine_first(df["hdd_anom"])
    except Exception as e:
        print("[WARN] HDD anomaly fetch failed: {}".format(e))

    # 4) ECB FX (EUR per USD)
    try:
        fx = fetch_ecb_eur_usd(start_date, end_date).set_index("date").sort_index()
        df["eur_per_usd"] = df.index.to_series().map(fx["eur_per_usd"]).ffill().bfill()
    except Exception as e:
        print("[WARN] ECB FX fetch failed: {}".format(e))

    # 5) Brent from EIA (preferred) or Yahoo (fallback)
    br_loaded = False
    eia_key = os.getenv("EIA_API_KEY", "").strip()
    if eia_key:
        try:
            br = fetch_eia_brent(start_date, end_date, eia_key).set_index("date").sort_index()
            # prefer live over synthetic via join+combine_first
            df = df.join(br["brent_usd_bbl"].rename("brent_live"), how="left")
            df["brent_usd_bbl"] = df["brent_live"].combine_first(df["brent_usd_bbl"])
            df.drop(columns=["brent_live"], inplace=True)
            br_loaded = True
        except Exception as e:
            print("[WARN] EIA Brent fetch failed: {}".format(e))

    if not br_loaded:
        try:
            br_yf = fetch_yahoo_brent(start_date, end_date).set_index("date").sort_index()
            df = df.join(br_yf["brent_usd_bbl"].rename("brent_yf"), how="left")
            df["brent_usd_bbl"] = df["brent_yf"].combine_first(df["brent_usd_bbl"])
            df.drop(columns=["brent_yf"], inplace=True)
            print("-- Brent crude oil --")
            print(df)
            br_loaded = True
        except Exception as e:
            print("[WARN] Yahoo Brent fetch failed: {}".format(e))

    # 6) Optional market feeds via CSV or Nasdaq Data Link
    ndl_key = get_ndl_api_key()

    # TTF
    ttf_csv = os.getenv("TTF_SOURCE_CSV", "").strip()
    if ttf_csv and os.path.exists(ttf_csv):
        dc = os.getenv("TTF_CSV_DATE_COL", "date"); vc = os.getenv("TTF_CSV_VALUE_COL", "value")
        try:
            ts = load_csv_series(ttf_csv, dc, vc).set_index("date")
            df["ttf_eur_mwh"] = df.index.to_series().map(ts["value"]).combine_first(df["ttf_eur_mwh"])  # prefer provided
        except Exception as e:
            print("[WARN] TTF CSV load failed: {}".format(e))
    elif ndl_key and os.getenv("NDL_TTF_CODE", "").strip():
        try:
            ts = fetch_ndl_timeseries(os.getenv("NDL_TTF_CODE").strip(), ndl_key, start_date, end_date).set_index("date")
            df["ttf_eur_mwh"] = df.index.to_series().map(ts["value"]).combine_first(df["ttf_eur_mwh"])  # prefer live
        except Exception as e:
            print("[WARN] NDL TTF fetch failed: {}".format(e))
    else:
        # Yahoo Finance fallback (no key, no manual download)
        try:
            ts = fetch_yahoo_ttf(start_date, end_date).set_index("date").sort_index()
            # JOIN instead of map to avoid 2-D shape issues
            df = df.join(ts["value"].rename("ttf_yf"), how="left")
            df["ttf_eur_mwh"] = df["ttf_yf"].combine_first(df["ttf_eur_mwh"])
            df.drop(columns=["ttf_yf"], inplace=True)
            print("--- TTF EUR/MWh ---")
            print(df)
        except Exception as e:
            print("[WARN] Yahoo TTF fetch failed: {}".format(e))

    # EUA
    eua_csv = os.getenv("EUA_SOURCE_CSV", "").strip()
    if eua_csv and os.path.exists(eua_csv):
        dc = os.getenv("EUA_CSV_DATE_COL", "date"); vc = os.getenv("EUA_CSV_VALUE_COL", "value")
        try:
            ts = load_csv_series(eua_csv, dc, vc).set_index("date")
            df["eua_eur_t"] = df.index.to_series().map(ts["value"]).combine_first(df["eua_eur_t"])  # prefer provided
        except Exception as e:
            print("[WARN] EUA CSV load failed: {}".format(e))
    elif ndl_key and os.getenv("NDL_EUA_CODE", "").strip():
        try:
            ts = fetch_ndl_timeseries(os.getenv("NDL_EUA_CODE").strip(), ndl_key, start_date, end_date).set_index("date")
            df["eua_eur_t"] = df.index.to_series().map(ts["value"]).combine_first(df["eua_eur_t"])  # prefer live
        except Exception as e:
            print("[WARN] NDL EUA fetch failed: {}".format(e))

    # JKM
    jkm_csv = os.getenv("JKM_SOURCE_CSV", "").strip()
    if jkm_csv and os.path.exists(jkm_csv):
        dc = os.getenv("JKM_CSV_DATE_COL", "date"); vc = os.getenv("JKM_CSV_VALUE_COL", "value")
        try:
            ts = load_csv_series(jkm_csv, dc, vc).set_index("date")
            df["jkm_usd_mmbtu"] = df.index.to_series().map(ts["value"]).combine_first(df["jkm_usd_mmbtu"])  # prefer provided
        except Exception as e:
            print("[WARN] JKM CSV load failed: {}".format(e))
    elif ndl_key and os.getenv("NDL_JKM_CODE", "").strip():
        try:
            ts = fetch_ndl_timeseries(os.getenv("NDL_JKM_CODE").strip(), ndl_key, start_date, end_date).set_index("date")
            df["jkm_usd_mmbtu"] = df.index.to_series().map(ts["value"]).combine_first(df["jkm_usd_mmbtu"])  # prefer live
        except Exception as e:
            print("[WARN] NDL JKM fetch failed: {}".format(e))

    return df

# -----------------------------
# Feature Engineering (Daily → Weekly)
# -----------------------------

FEATURE_COLUMNS_BASE = [
    "ttf_eur_mwh",
    "jkm_usd_mmbtu",
    "storage_fill_pct",
    "storage_dev_5y_pp",
    "norway_outage",
    "wind_index",
    "hdd_anom",
    "eua_eur_t",
    "brent_usd_bbl",
]

def _weekly_aggregate(df: pd.DataFrame) -> pd.DataFrame:
    agg_map = {
        "ttf_eur_mwh": "mean",
        "jkm_usd_mmbtu": "mean",
        "storage_fill_pct": "mean",
        "storage_dev_5y_pp": "mean",
        "norway_outage": "max",
        "wind_index": "mean",
        "hdd_anom": "mean",
        "eua_eur_t": "mean",
        "brent_usd_bbl": "mean",
    }
    if "eur_per_usd" in df.columns:
        agg_map["eur_per_usd"] = "mean"
    weekly = df.resample("W-FRI").agg(agg_map)
    weekly.index.name = "week_end"
    return weekly

def _add_derived_features(weekly: pd.DataFrame) -> pd.DataFrame:
    out = weekly.copy()
    # Unit-safe JKM conversion and TTF–JKM spread
    has_fx_col = "eur_per_usd" in out.columns and out["eur_per_usd"].notna().any()
    fx_env = os.getenv("EUR_PER_USD", "").strip()
    fx_env_val = float(fx_env) if fx_env else None
    if has_fx_col:
        out["jkm_eur_mwh"] = (out["jkm_usd_mmbtu"] * out["eur_per_usd"]) / 0.29307107
    elif fx_env_val is not None:
        out["jkm_eur_mwh"] = (out["jkm_usd_mmbtu"] * fx_env_val) / 0.29307107
    else:
        out["jkm_eur_mwh"] = np.nan
    out["ttf_minus_jkm"] = out["ttf_eur_mwh"] - out["jkm_eur_mwh"] if out["jkm_eur_mwh"].notna().any() else np.nan

    # WoW diffs & 4-week MAs
    for col in [
        "ttf_eur_mwh","jkm_usd_mmbtu","jkm_eur_mwh","storage_fill_pct","storage_dev_5y_pp",
        "wind_index","hdd_anom","eua_eur_t","brent_usd_bbl","ttf_minus_jkm","eur_per_usd",
    ]:
        if col in out.columns:
            out["{c}_wow".format(c=col)] = out[col].diff()
            out["{c}_ma4".format(c=col)] = out[col].rolling(4).mean()
    return out

def build_weekly_dataset(daily: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) weekly dataset with one-step lookahead alignment.

    Target y = next week’s return of TTF vs current week average:
        y_t = (avg_TTF_{t+1} / avg_TTF_{t}) - 1

    Features X = engineered weekly metrics shifted by 1 week (predict t+1 with info up to t).
    """
    weekly = _weekly_aggregate(daily)

    # Target: next-week return
    ttf = weekly["ttf_eur_mwh"].copy()
    next_week_avg = ttf.shift(-1)
    y = (next_week_avg / ttf) - 1.0

    # Engineered features at weekly level
    X_raw = _add_derived_features(weekly)

    # Shift features by 1 week to ensure strict causality
    X = X_raw.shift(1)

    # Robustness to missing engineered cols
    X = X.dropna(axis=1, how="all").ffill().bfill()

    # Align X and y and drop only rows with missing y
    valid = y.notna()
    X = X.loc[valid]
    y = y.loc[valid]

    # Safety: drop any remaining rows with NaNs
    mask_no_nan = X.notna().all(axis=1)
    X = X.loc[mask_no_nan]
    y = y.loc[mask_no_nan]

    return X, y

# -----------------------------
# Modeling
# -----------------------------

def train_ridge(X: pd.DataFrame, y: pd.Series, alpha: float | None = None) -> Tuple[Pipeline, Dict[str, float]]:
    """Train StandardScaler + Ridge with optional TS-CV alpha tuning.

    If `alpha` is None, pick it via walk-forward CV over a small grid.
    Returns the fitted pipeline and a metrics dict (incl. chosen alpha).
    """
    n = len(X)
    if n < 30:
        raise ValueError("Not enough samples to train")

    # Hyperparam grid (kept tiny for speed)
    grid = [0.05, 0.1, 0.3, 1.0, 3.0, 10.0] if alpha is None else [alpha]

    best_alpha = None
    best_score = -np.inf

    tscv = TimeSeriesSplit(n_splits=min(5, max(2, n // 30)))

    for a in grid:
        scores = []
        for tr, va in tscv.split(X):
            pipe_cv = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=a))])
            pipe_cv.fit(X.iloc[tr], y.iloc[tr])
            yhat = pipe_cv.predict(X.iloc[va])
            scores.append(r2_score(y.iloc[va], yhat))
        mean_score = float(np.mean(scores)) if scores else -np.inf
        if mean_score > best_score:
            best_score = mean_score
            best_alpha = a

    # Final fit using best alpha
    pipe = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=best_alpha))])

    split = int(n * 0.8)
    X_train, X_val = X.iloc[:split], X.iloc[split:]
    y_train, y_val = y.iloc[:split], y.iloc[split:]
    pipe.fit(X_train, y_train)

    y_hat = pipe.predict(X_val)
    metrics = {
        "alpha": float(best_alpha),
        "r2": float(r2_score(y_val, y_hat)),
        "mae": float(mean_absolute_error(y_val, y_hat)),
        "rmse": float(np.sqrt(mean_squared_error(y_val, y_hat))),
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "tscv_mean": float(best_score),
    }

    # Add TS-CV std for the chosen alpha
    scores = []
    for tr, va in tscv.split(X):
        pipe.fit(X.iloc[tr], y.iloc[tr])
        yhat = pipe.predict(X.iloc[va])
        scores.append(r2_score(y.iloc[va], yhat))
    if scores:
        metrics["tscv_std"] = float(np.std(scores))

    return pipe, metrics

def predict_next_week(pipe: Pipeline, X: pd.DataFrame) -> Tuple[float, pd.Timestamp]:
    if X.empty:
        raise ValueError("Feature matrix X is empty")
    last_dt = X.index[-1]
    pred = float(pipe.predict(X.iloc[[-1]])[0])
    return pred, last_dt

def classify_direction(pred_ret: float, thr: float = 0.01) -> str:
    return "UP" if pred_ret >= thr else ("DOWN" if pred_ret <= -thr else "FLAT")

def linear_contributions(pipe: Pipeline, X_row: pd.DataFrame) -> List[Tuple[str, float]]:
    scaler: StandardScaler = pipe.named_steps["scaler"]
    model = pipe.named_steps.get("ridge") or pipe.named_steps.get("logit") or pipe.named_steps.get("logistic")
    if model is None or not hasattr(model, "coef_"):
        return []
    z = (X_row.values - scaler.mean_) / scaler.scale_
    beta = model.coef_[0] if getattr(model.coef_, "ndim", 1) == 2 else model.coef_
    contrib = (z * beta).ravel()
    pairs = list(zip(X_row.columns.tolist(), contrib.tolist()))
    pairs.sort(key=lambda x: abs(x[1]), reverse=True)
    return pairs

# -----------------------------
# Reporting (string-safe)
# -----------------------------

def make_report(
    pred_ret: float,
    asof_week: pd.Timestamp,
    direction: str,
    metrics: Dict[str, float],
    top_contribs: List[Tuple[str, float]],
    n_list: int = 8,
) -> str:
    """Build the markdown report using only single-line strings joined at the end."""
    pct = pred_ret * 100.0
    week_str = asof_week.strftime("%Y-%m-%d")
    arrow = {"UP": "↑", "DOWN": "↓", "FLAT": "→"}.get(direction, "→")

    header_line = "# Weekly Gas Price Outlook (as of {week})".format(week=week_str)

    if metrics.get("mode") == "classify":
        trend_line = "**Trend**: {d} {a}".format(d=direction, a=arrow)
    else:
        trend_line = "**Trend**: {d} {a}  |  **Expected change**: {pct:+.2f}% (1-week ahead)".format(d=direction, a=arrow, pct=pct)

    model_title = "## Model Snapshot"
    if metrics.get("mode") == "classify":
        tmean = metrics.get("tscv_mean", float("nan"))
        tstd  = metrics.get("tscv_std",  float("nan"))
        model_line = "Logistic (UP/DOWN)  |  TS-CV Accuracy: {acc:.3f} ± {std:.3f}".format(acc=tmean, std=tstd)
    else:
        r2   = metrics.get("r2",   float("nan"))
        mae  = metrics.get("mae",  float("nan"))
        rmse = metrics.get("rmse", float("nan"))
        ntr  = metrics.get("n_train", 0)
        nva  = metrics.get("n_val",   0)
        tmean = metrics.get("tscv_mean")
        tstd  = metrics.get("tscv_std")
        alpha_used = metrics.get("alpha", 1.0)
        model_line = (
            "Ridge(alpha={alpha:.2f})  |  R²: {r2:.3f}  |  MAE: {mae:.4f}  |  RMSE: {rmse:.4f}  | "
            "Train/Val weeks: {ntr}/{nva}"
        ).format(alpha=alpha_used, r2=r2, mae=mae, rmse=rmse, ntr=ntr, nva=nva)
        if tmean is not None:
            model_line += "  |  TS-CV R²: {m:.3f} ± {s:.3f}".format(m=tmean, s=tstd)

    contrib_title = "## Top Factor Contributions (standardized units)"
    if top_contribs:
        contrib_lines = "\n".join(["- {n}: {v:+.3f}".format(n=n, v=v) for n, v in top_contribs[:n_list]])
    else:
        contrib_lines = "- (no contributions computed)"

    note_line = (
        "*Note:* Contributions are approximations from standardized features and linear coefficients; "
        "they provide **directional** insight, not exact price impacts."
    )
    hr_line = "---"
    impl_line = (
        "**Implementation status:** Live storage/wind/FX/HDD; optional Brent via EIA; "
        "TTF/EUA/JKM via NDL or CSV if provided."
    )

    parts = [
        header_line,
        trend_line,
        "",
        model_title,
        model_line,
        "",
        contrib_title,
        contrib_lines,
        "",
        note_line,
        "",
        hr_line,
        impl_line,
    ]
    return "\n".join(parts) + "\n"

# -----------------------------
# CLI
# -----------------------------

def run_pipeline(args: argparse.Namespace) -> None:
    cfg = GasWatchConfig(direction_threshold=args.direction_threshold, ridge_alpha=1.0, min_weeks_to_train=args.min_weeks_to_train, seed=args.seed)
    daily = load_or_synthesize_daily_data(n_days=args.days, seed=cfg.seed)
    X, y = build_weekly_dataset(daily)
    if len(X) < cfg.min_weeks_to_train:
        raise RuntimeError("Not enough history to train: have {} weekly rows, need {}.".format(len(X), cfg.min_weeks_to_train))

    if args.mode == "classify":
        def build_direction_labels(y: pd.Series, thr: float = 0.005) -> pd.Series:
            lab = pd.Series(index=y.index, dtype="object")
            lab[y >= thr] = "UP"; lab[y <= -thr] = "DOWN"; lab[(y > -thr) & (y < thr)] = "FLAT"
            return lab
        y_dir = build_direction_labels(y, thr=0.005)
        keep = y_dir.isin(["UP", "DOWN"])  # drop FLAT for boundary clarity
        Xc, yc = X.loc[keep], y_dir.loc[keep]
        if len(Xc) < cfg.min_weeks_to_train:
            raise RuntimeError("Not enough history to train (classify): have {}, need {}".format(len(Xc), cfg.min_weeks_to_train))
        clf = Pipeline([("scaler", StandardScaler()), ("logit", LogisticRegression(max_iter=500))])
        tscv = TimeSeriesSplit(n_splits=min(5, max(2, len(Xc) // 30)))
        accs = []
        for tr, va in tscv.split(Xc):
            clf.fit(Xc.iloc[tr], yc.iloc[tr]); yhat = clf.predict(Xc.iloc[va]); accs.append((yc.iloc[va] == yhat).mean())
        metrics = {"mode": "classify", "tscv_mean": float(np.mean(accs)), "tscv_std": float(np.std(accs))}
        clf.fit(Xc, yc)
        x_last = X.iloc[[-1]]; pred_label = clf.predict(x_last)[0]
        contribs = linear_contributions(clf, x_last)
        report_md = make_report(0.0, X.index[-1], pred_label, metrics, contribs, n_list=args.topk)
        print(report_md)
        payload = {"asof_week": X.index[-1].strftime("%Y-%m-%d"), "direction": pred_label, "metrics": metrics}
    else:
        pipe, metrics = train_ridge(X, y, alpha=None)
        pred_ret, asof_week = predict_next_week(pipe, X)
        hist_abs = float(np.abs(y).quantile(0.80)) if len(y) else 0.0
        cap = hist_abs if hist_abs > 0 else 0.08
        pred_ret = float(np.clip(pred_ret, -cap, +cap))
        direction = classify_direction(pred_ret, cfg.direction_threshold)
        contribs = linear_contributions(pipe, X.iloc[[-1]])
        report_md = make_report(pred_ret, asof_week, direction, metrics, contribs, n_list=args.topk)
        print(report_md)
        payload = {"asof_week": asof_week.strftime("%Y-%m-%d"), "predicted_return": pred_ret, "direction": direction, "metrics": metrics, "top_contributions": contribs[: args.topk]}

    if args.report_out:
        Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report_out).write_text(report_md, encoding="utf-8")
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Weekly European Gas Price Forecaster")
    p.add_argument("--run-all", action="store_true", help="Run the full pipeline end-to-end")
    p.add_argument("--days", type=int, default=3 * 365, help="How many daily observations to load/synthesize")
    p.add_argument("--direction-threshold", type=float, default=0.01, help="Directional threshold (e.g., 0.01 = 1%)")
    p.add_argument("--min-weeks-to-train", type=int, default=60, help="Minimum weeks needed to train")
    p.add_argument("--seed", type=int, default=42, help="Random seed for synthetic demo data")
    p.add_argument("--topk", type=int, default=8, help="How many top contributions to show")
    p.add_argument("--report-out", type=str, default="", help="Path to save the Markdown report")
    p.add_argument("--json-out", type=str, default="", help="Path to save the JSON output")
    p.add_argument("--mode", choices=["regress", "classify"], default="regress", help="Weekly forecast mode: regress (pct) or classify (direction)")
    return p

def main() -> None:
    args = build_arg_parser().parse_args()
    if not args.run_all:
        print("Nothing to do: pass --run-all. Example:\n\n  python gas_watch_weekly.py --run-all \\\n  --report-out ./output/weekly_gas_watch.md \\\n  --json-out ./output/weekly_gas_watch.json\n")
        return
    run_pipeline(args)

if __name__ == "__main__":
    main()

