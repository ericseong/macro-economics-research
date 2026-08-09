#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
national_pension_cer.py

Fetch USD/KRW daily close exchange rate for the last N years via yfinance,
compute mean and standard deviation, then compute CER (Critical Exchange Rate):

    CER_UP = mean + k * std
    CER_LO = mean - k * std

Plot the daily close series and CER lines using Plotly.

Usage example:
  national_pension_cer.py --years 10 --k 2.58
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import sys

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    print("[ERROR] yfinance is not installed. Try: pip install yfinance", file=sys.stderr)
    raise

try:
    import plotly.graph_objects as go
except ImportError:
    print("[ERROR] plotly is not installed. Try: pip install plotly", file=sys.stderr)
    raise


DEFAULT_TICKER = "USDKRW=X"  # Yahoo Finance FX ticker for USD/KRW


@dataclass(frozen=True)
class CerResult:
    ticker: str
    years: float
    start: pd.Timestamp
    end: pd.Timestamp
    n_points: int
    mean: float
    std: float
    cer_up: float
    cer_lo: float
    series: pd.Series


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute CER (Critical Exchange Rate) from USD/KRW daily close via yfinance and plot it with Plotly."
    )
    p.add_argument(
        "--years",
        type=float,
        required=True,
        help="Lookback window in years (e.g., 1, 3, 5).",
    )
    p.add_argument(
        "--ticker",
        type=str,
        default=DEFAULT_TICKER,
        help=f"Yahoo Finance ticker. Default: {DEFAULT_TICKER}",
    )
    p.add_argument(
        "--k",
        type=float,
        default=2.58,
        help="Multiplier for std in CER formula. Default: 2.58",
    )
    p.add_argument(
        "--output",
        type=str,
        default="",
        help="If set, save chart to this HTML file path (e.g., out.html).",
    )
    p.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open the interactive chart window.",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Print debug information to help diagnose data/plot issues.",
    )
    return p.parse_args()


def dbg(enabled: bool, msg: str) -> None:
    if enabled:
        print(msg)


def _to_utc_date_str(ts: datetime) -> str:
    return ts.astimezone(timezone.utc).strftime("%Y-%m-%d")


def _select_price_series(df: pd.DataFrame, ticker: str, debug: bool) -> pd.Series:
    dbg(debug, f"[DEBUG] Raw df.columns type: {type(df.columns)}")
    dbg(debug, f"[DEBUG] Raw df.columns: {df.columns}")

    if isinstance(df.columns, pd.MultiIndex):
        for field in ("Adj Close", "Close"):
            key = (field, ticker)
            if key in df.columns:
                s = df[key].copy()
                dbg(debug, f"[DEBUG] Selected MultiIndex column: {key}")
                return s

        for field in ("Adj Close", "Close"):
            if field in df.columns.get_level_values(0):
                sub = df[field]
                dbg(debug, f"[DEBUG] MultiIndex detected; df['{field}'] columns: {list(sub.columns)}")
                if isinstance(sub, pd.DataFrame) and sub.shape[1] == 1:
                    s = sub.iloc[:, 0].copy()
                    dbg(debug, f"[DEBUG] Auto-selected only column under field '{field}': {sub.columns[0]}")
                    return s

        raise RuntimeError(
            f"Could not find ('Adj Close' or 'Close') for ticker '{ticker}' in MultiIndex columns."
        )

    col = "Adj Close" if "Adj Close" in df.columns else ("Close" if "Close" in df.columns else None)
    if col is None:
        raise RuntimeError(f"Downloaded data does not contain Close/Adj Close columns. Columns: {list(df.columns)}")

    dbg(debug, f"[DEBUG] Selected SingleIndex column: {col}")
    return df[col].copy()


def fetch_series(ticker: str, years: float, debug: bool) -> pd.Series:
    if years <= 0:
        raise ValueError("--years must be > 0")

    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=int(years * 365.25) + 10)

    dbg(debug, f"[DEBUG] Request ticker={ticker}, years={years}")
    dbg(debug, f"[DEBUG] Download start={_to_utc_date_str(start_dt)}, end={_to_utc_date_str(end_dt)}")

    df = yf.download(
        tickers=ticker,
        start=_to_utc_date_str(start_dt),
        end=_to_utc_date_str(end_dt + timedelta(days=1)),
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=True,
        group_by="column",
    )

    if df is None or df.empty:
        raise RuntimeError(f"No data returned from yfinance for ticker: {ticker}")

    dbg(debug, f"[DEBUG] Downloaded df shape={df.shape}")
    dbg(debug, f"[DEBUG] df.head(3):\n{df.head(3)}")
    dbg(debug, f"[DEBUG] df.tail(3):\n{df.tail(3)}")
    dbg(debug, f"[DEBUG] df.index type={type(df.index)}, tz={getattr(df.index, 'tz', None)}")

    s = _select_price_series(df, ticker, debug=debug)

    dbg(debug, f"[DEBUG] Extracted series type={type(s)}, name={s.name}, dtype={s.dtype}")
    dbg(debug, f"[DEBUG] Series raw head(5):\n{s.head(5)}")
    dbg(debug, f"[DEBUG] Series raw tail(5):\n{s.tail(5)}")

    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index, errors="coerce")
    s = s.dropna()

    s = s.sort_index()
    if s.index.has_duplicates:
        dup_cnt = int(s.index.duplicated().sum())
        dbg(debug, f"[DEBUG] Found duplicated timestamps: {dup_cnt}. Keeping last occurrence.")
        s = s[~s.index.duplicated(keep="last")]

    end_ts = s.index.max()
    start_cut = end_ts - pd.Timedelta(days=years * 365.25)
    s = s.loc[s.index >= start_cut]

    dbg(debug, f"[DEBUG] After trimming: points={len(s)}, start={s.index.min()}, end={s.index.max()}")
    dbg(debug, f"[DEBUG] NaN ratio after cleaning: {float(s.isna().mean()):.6f}")
    if len(s) > 0:
        dbg(debug, f"[DEBUG] Value range: min={float(s.min()):.6f}, max={float(s.max()):.6f}")
        dbg(debug, f"[DEBUG] Last value: {float(s.iloc[-1]):.6f}")

    if s.empty:
        raise RuntimeError("Series is empty after applying the lookback window. Try a larger --years value.")

    return s


def compute_cer(series: pd.Series, ticker: str, years: float, k: float, debug: bool) -> CerResult:
    vals = series.values.astype(float)

    mean = float(np.mean(vals))
    std = float(np.std(vals, ddof=1))  # sample std
    cer_up = mean + k * std
    cer_lo = mean - k * std

    dbg(debug, f"[DEBUG] mean={mean}, std(ddof=1)={std}, k={k}, cer_up={cer_up}, cer_lo={cer_lo}")

    return CerResult(
        ticker=ticker,
        years=years,
        start=pd.Timestamp(series.index.min()),
        end=pd.Timestamp(series.index.max()),
        n_points=int(series.shape[0]),
        mean=mean,
        std=std,
        cer_up=cer_up,
        cer_lo=cer_lo,
        series=series,
    )


def make_plot(res: CerResult, k: float, debug: bool) -> go.Figure:
    s = res.series
    x = s.index
    y = s.values

    # --- y-range MUST include: series range + CER_UP/CER_LO + mean (+/- std too, for safety) ---
    candidates = np.array([
        float(np.nanmin(y)),
        float(np.nanmax(y)),
        res.cer_up,
        res.cer_lo,
        res.mean,
        res.mean + res.std,
        res.mean - res.std,
    ], dtype=float)

    y_min = float(np.nanmin(candidates))
    y_max = float(np.nanmax(candidates))

    pad = (y_max - y_min) * 0.05 if y_max > y_min else max(1.0, abs(y_min) * 0.01)
    y_range = [y_min - pad, y_max + pad]

    dbg(debug, f"[DEBUG] Plot y_range={y_range} (min={y_min}, max={y_max}, pad={pad})")
    dbg(debug, f"[DEBUG] mean={res.mean}, std={res.std}, cer_up={res.cer_up}, cer_lo={res.cer_lo}")

    fig = go.Figure()

    # Daily close
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            name=f"{res.ticker} Daily Close",
            hovertemplate="Date=%{x|%Y-%m-%d}<br>USDKRW=%{y:.2f}<extra></extra>",
        )
    )

    # Mean line
    fig.add_trace(
        go.Scatter(
            x=[x.min(), x.max()],
            y=[res.mean, res.mean],
            mode="lines",
            name="Mean",
            hovertemplate="Mean=%{y:.2f}<extra></extra>",
        )
    )

    # Std lines (mean ± 1 std)
    fig.add_trace(
        go.Scatter(
            x=[x.min(), x.max()],
            y=[res.mean + res.std, res.mean + res.std],
            mode="lines",
            name="+1σ",
            hovertemplate="+1σ=%{y:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[x.min(), x.max()],
            y=[res.mean - res.std, res.mean - res.std],
            mode="lines",
            name="-1σ",
            hovertemplate="-1σ=%{y:.2f}<extra></extra>",
        )
    )

    # CER lines (mean ± k·std)
    fig.add_trace(
        go.Scatter(
            x=[x.min(), x.max()],
            y=[res.cer_up, res.cer_up],
            mode="lines",
            name=f"CER_UP = mean + {k}·std",
            hovertemplate="CER_UP=%{y:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[x.min(), x.max()],
            y=[res.cer_lo, res.cer_lo],
            mode="lines",
            name=f"CER_LO = mean - {k}·std",
            hovertemplate="CER_LO=%{y:.2f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=(
            f"CER (Critical Exchange Rate) from {res.ticker} — last {res.years:g} years<br>"
            f"mean={res.mean:.2f}, std={res.std:.2f}, "
            f"CER_UP={res.cer_up:.2f}, CER_LO={res.cer_lo:.2f} (n={res.n_points})"
        ),
        xaxis_title="Date",
        yaxis_title="USD to KRW (Daily Close)",
        hovermode="x unified",
        legend_title="Series",
    )

    fig.update_yaxes(range=y_range)

    # annotate at right edge
    fig.add_annotation(x=x.max(), y=res.cer_up, xref="x", yref="y",
                       text=f"CER_UP {res.cer_up:.2f}", showarrow=True, arrowhead=2, ax=-40, ay=-20)
    fig.add_annotation(x=x.max(), y=res.cer_lo, xref="x", yref="y",
                       text=f"CER_LO {res.cer_lo:.2f}", showarrow=True, arrowhead=2, ax=-40, ay=20)
    fig.add_annotation(x=x.max(), y=res.mean, xref="x", yref="y",
                       text=f"Mean {res.mean:.2f}", showarrow=True, arrowhead=2, ax=-40, ay=-40)

    return fig


def main() -> int:
    args = _parse_args()

    try:
        series = fetch_series(args.ticker, args.years, debug=args.debug)
        res = compute_cer(series, args.ticker, args.years, args.k, debug=args.debug)

        print("=== CER Calculation ===")
        print(f"Ticker: {res.ticker}")
        print(f"Period: {res.start.date()} ~ {res.end.date()}")
        print(f"Data points: {res.n_points}")
        print(f"Mean (USDKRW): {res.mean:.6f}")
        print(f"Std  (USDKRW): {res.std:.6f}  (sample std, ddof=1)")
        print(f"CER_UP = mean + {args.k} * std = {res.cer_up:.6f}")
        print(f"CER_LO = mean - {args.k} * std = {res.cer_lo:.6f}")

        fig = make_plot(res, args.k, debug=args.debug)

        if args.output:
            fig.write_html(args.output, include_plotlyjs="cdn")
            print(f"[INFO] Saved chart to: {args.output}")

        if not args.no_show:
            fig.show()

        return 0

    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

