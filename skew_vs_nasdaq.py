#!/usr/bin/env python3
"""Plot the CBOE SKEW Index against the Nasdaq Composite Index."""

import argparse
from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from plotly.subplots import make_subplots


TICKERS = {
    "skew": "^SKEW",
    "nasdaq": "^IXIC",  # Nasdaq Composite Index
}


def positive_years(value: str) -> int:
    """Parse --years and reject zero or negative values."""
    try:
        years = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("years must be a positive integer") from exc
    if years < 1:
        raise argparse.ArgumentTypeError("years must be a positive integer")
    return years


def download_history(ticker: str, start: date, end: date) -> pd.DataFrame:
    """Download daily history, handling yfinance's empty response."""
    history = yf.Ticker(ticker).history(
        start=start.isoformat(),
        end=end.isoformat(),  # yfinance treats end as exclusive
        interval="1d",
        auto_adjust=False,
        actions=False,
    )
    if history.empty or "Close" not in history:
        raise RuntimeError(f"No price data returned for {ticker}.")

    # yfinance may return a timezone-aware index; Plotly works consistently
    # with the daily date labels once the timezone is removed.
    if history.index.tz is not None:
        history.index = history.index.tz_localize(None)
    return history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare the SKEW Index and Nasdaq Composite in one chart."
    )
    parser.add_argument(
        "--years",
        type=positive_years,
        default=3,
        help="number of recent years to download (default: 3)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    end = date.today() + timedelta(days=1)
    start = (pd.Timestamp(end) - pd.DateOffset(years=args.years)).date()

    skew = download_history(TICKERS["skew"], start, end)["Close"].rename(
        TICKERS["skew"]
    )
    nasdaq_history = download_history(TICKERS["nasdaq"], start, end)
    ohlc_columns = ["Open", "High", "Low", "Close"]
    missing_columns = [
        column for column in ohlc_columns + ["Volume"] if column not in nasdaq_history
    ]
    if missing_columns:
        raise RuntimeError(
            "Nasdaq data is missing required columns: " + ", ".join(missing_columns)
        )
    nasdaq_ohlc = nasdaq_history[ohlc_columns].rename(
        columns={
            "Open": "Nasdaq Open",
            "High": "Nasdaq High",
            "Low": "Nasdaq Low",
            "Close": "Nasdaq Close",
        }
    )
    volume = nasdaq_history["Volume"].rename("Nasdaq Volume")
    prices = pd.concat([skew, nasdaq_ohlc, volume], axis=1, join="inner").dropna(
        subset=[TICKERS["skew"], "Nasdaq Open", "Nasdaq High", "Nasdaq Low", "Nasdaq Close"]
    )
    prices["Nasdaq Volume"] = prices["Nasdaq Volume"].fillna(0)
    if prices.empty:
        raise RuntimeError("The SKEW and Nasdaq series have no overlapping dates.")

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.035,
        row_heights=[0.74, 0.26],
        specs=[[{"secondary_y": True}], [{}]],
    )
    fig.add_trace(
        go.Scatter(
            x=prices.index,
            y=prices[TICKERS["skew"]],
            name="CBOE SKEW Index",
            mode="lines",
            line={"color": "red", "width": 1.7},
        ),
        row=1,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Candlestick(
            x=prices.index,
            open=prices["Nasdaq Open"],
            high=prices["Nasdaq High"],
            low=prices["Nasdaq Low"],
            close=prices["Nasdaq Close"],
            name="Nasdaq Composite",
            increasing_line_color="#1565C0",
            decreasing_line_color="#D32F2F",
            increasing_fillcolor="#1565C0",
            decreasing_fillcolor="#D32F2F",
        ),
        row=1,
        col=1,
        secondary_y=True,
    )
    fig.add_trace(
        go.Bar(
            x=prices.index,
            y=prices["Nasdaq Volume"],
            name="Nasdaq Volume",
            marker_color="rgba(70, 130, 180, 0.55)",
        ),
        row=2,
        col=1,
    )

    skew_min = min(100, float(prices[TICKERS["skew"]].min()))
    skew_max = max(150, float(prices[TICKERS["skew"]].max()))
    pad = max((skew_max - skew_min) * 0.03, 1)
    fig.update_yaxes(
        title_text="SKEW Index",
        title_font_color="red",
        tickfont_color="red",
        range=[skew_min - pad, skew_max + pad],
        row=1,
        col=1,
        secondary_y=False,
    )
    fig.update_yaxes(
        title_text="Nasdaq Composite Index",
        title_font_color="blue",
        tickfont_color="blue",
        row=1,
        col=1,
        secondary_y=True,
    )
    fig.add_hrect(
        y0=100,
        y1=150,
        fillcolor="gray",
        opacity=0.16,
        line_width=0,
        layer="below",
        annotation_text="SKEW 100–150",
        annotation_position="top left",
        row=1,
        col=1,
    )
    fig.update_yaxes(title_text="Volume", row=2, col=1, rangemode="tozero")
    fig.update_yaxes(tickformat=".2s", row=2, col=1)
    fig.update_layout(
        title=f"SKEW Index vs Nasdaq Composite ({args.years} year(s))",
        xaxis2_title="Date",
        hovermode="x unified",
        template="plotly_white",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        margin={"l": 75, "r": 85, "t": 95, "b": 60},
    )
    fig.update_xaxes(showgrid=True, rangeslider_visible=False, row=1, col=1)
    fig.update_xaxes(
        showgrid=True,
        rangeslider={"visible": True, "thickness": 0.12},
        row=2,
        col=1,
    )
    fig.show(config={"scrollZoom": True})


if __name__ == "__main__":
    main()
