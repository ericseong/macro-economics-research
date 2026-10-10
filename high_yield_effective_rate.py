import argparse
import os
from datetime import datetime, timedelta

import pandas as pd
from fredapi import Fred
import yfinance as yf
import plotly.graph_objects as go


FRED_SERIES = {
    "effective_yield": "BAMLH0A0HYM2EY",
    "credit_spread": "BAMLH0A0HYM2",
    "treasury_rate_3y": "DGS3",
    "treasury_rate_5y": "DGS5",
}


def load_fred_api_key():
    try:
        with open("./credentials/credential_fred_api.txt", "r") as file:
            return file.read().strip()
    except FileNotFoundError:
        raise RuntimeError(
            "FRED API key is missing! Store it in ./credentials/credential_fred_api.txt."
        )


def get_fred_series(fred: Fred, series_id: str, start_date: datetime):
    series = fred.get_series(series_id, observation_start=start_date)
    series.index = series.index.date
    return series


def get_spread_data(fred_api_key: str, years: int):
    """Return ICE BofA US High Yield Index Effective Yield (kept for compatibility)."""
    fred = Fred(api_key=fred_api_key)
    start_date = datetime.today() - timedelta(days=365 * years)
    return get_fred_series(fred, FRED_SERIES["effective_yield"], start_date)


def get_credit_spread_data(fred_api_key: str, years: int):
    """Return the ICE BofA US High Yield Index OAS series.

    FRED provides the high-yield OAS for the full index, rather than separate
    3-year and 5-year high-yield OAS series.
    """
    fred = Fred(api_key=fred_api_key)
    start_date = datetime.today() - timedelta(days=365 * years)
    return get_fred_series(
        fred, FRED_SERIES["credit_spread"], start_date
    ).rename("US High-Yield OAS")


def get_treasury_rate_data(fred_api_key: str, years: int):
    """Return the mean of the 3-year and 5-year Treasury constant-maturity rates."""
    fred = Fred(api_key=fred_api_key)
    start_date = datetime.today() - timedelta(days=365 * years)
    rate_3y = get_fred_series(fred, FRED_SERIES["treasury_rate_3y"], start_date)
    rate_5y = get_fred_series(fred, FRED_SERIES["treasury_rate_5y"], start_date)
    return pd.concat(
        [rate_3y.rename("3Y Treasury"), rate_5y.rename("5Y Treasury")], axis=1
    ).ffill().mean(axis=1).rename("Average Treasury Rate")


def get_spy_data(years: int):
    end = datetime.today()
    start = end - timedelta(days=365 * years)
    spy_df = yf.download(
        tickers="SPY", start=start, end=end, progress=False, auto_adjust=True
    )

    if isinstance(spy_df.columns, pd.MultiIndex):
        spy_ohlc = spy_df.xs("SPY", axis=1, level=1)[
            ["Open", "High", "Low", "Close"]
        ]
    else:
        spy_ohlc = spy_df[["Open", "High", "Low", "Close"]]

    spy_ohlc.index = spy_ohlc.index.date
    return spy_ohlc


def plot_combined(
    spread_series,
    spy_ohlc_raw,
    treasury_series=None,
    credit_spread_series=None,
):
    # Retain the original effective-yield dates as the chart's date index.
    x_dates = spread_series.index
    spread_series = spread_series.reindex(x_dates).ffill()
    spy_ohlc = spy_ohlc_raw.reindex(x_dates).ffill()
    spy_series = spy_ohlc["Close"]
    if treasury_series is not None:
        treasury_series = treasury_series.reindex(x_dates).ffill()
    if credit_spread_series is not None:
        credit_spread_series = credit_spread_series.reindex(x_dates).ffill()

    # Compute the original 10-day moving averages.
    yield_ma10 = spread_series.rolling(window=10).mean()
    spy_ma10 = spy_series.rolling(window=10).mean()

    print("DEBUG: Yield range:", spread_series.min(), "to", spread_series.max())
    print("DEBUG: SPY range:", spy_series.min(), "to", spy_series.max())
    print("DEBUG: Date range:", x_dates.min(), "to", x_dates.max())
    print("DEBUG: Series length:", len(spread_series), len(spy_series))

    fig = go.Figure()

    # High-yield effective yield (original series).
    fig.add_trace(go.Scatter(
        x=x_dates,
        y=spread_series.values,
        name="High-Yield Effective Yield (%)",
        yaxis="y1",
        line=dict(color="black", width=2),
    ))

    # Average of the 3-year and 5-year Treasury rates.
    if treasury_series is not None:
        fig.add_trace(go.Scatter(
            x=x_dates,
            y=treasury_series.values,
            name="US Treasury Rate Avg (3Y & 5Y, %)",
            yaxis="y1",
            line=dict(color="green", width=1.8),
        ))

    # Broad US high-yield index OAS (no maturity-specific HY OAS series in FRED).
    if credit_spread_series is not None:
        fig.add_trace(go.Scatter(
            x=x_dates,
            y=credit_spread_series.values,
            name="US High-Yield Credit Spread (OAS, %)",
            yaxis="y1",
            line=dict(color="red", width=1.8),
        ))

    # Original effective-yield MA10.
    fig.add_trace(go.Scatter(
        x=x_dates,
        y=yield_ma10.values,
        name="Yield MA10",
        yaxis="y1",
        line=dict(color="black", width=1.5, dash="dot"),
    ))

    # SPY candlesticks replace the original price line; MA10 remains unchanged.
    fig.add_trace(go.Candlestick(
        x=x_dates,
        open=spy_ohlc["Open"].values,
        high=spy_ohlc["High"].values,
        low=spy_ohlc["Low"].values,
        close=spy_ohlc["Close"].values,
        name="SPY Price",
        yaxis="y2",
        increasing_line_color="green",
        decreasing_line_color="red",
    ))
    fig.add_trace(go.Scatter(
        x=x_dates,
        y=spy_ma10.values,
        name="SPY MA10",
        yaxis="y2",
        line=dict(color="royalblue", width=1.5, dash="dot"),
    ))

    # Reference lines at 7% and 9% (original behavior).
    for level in [7, 9]:
        fig.add_hline(
            y=level,
            line=dict(color="gray", dash="dot"),
            annotation_text=f"{level}%",
            annotation_position="top right",
        )

    # Credit-spread reference level requested by the user.
    fig.add_hline(
        y=3,
        line=dict(color="red", dash="dot"),
        annotation_text="Credit Spread 3%",
        annotation_position="top right",
    )

    fig.update_layout(
        title="High-Yield Effective Yield vs. HY OAS and SPY (with MA10)",
        dragmode="pan",
        xaxis=dict(
            title="Date",
            rangeslider=dict(visible=True),
            type="date",
        ),
        yaxis=dict(
            title=dict(text="Yield / Rate / Spread (%)", font=dict(color="firebrick")),
            side="left",
            showgrid=True,
            showline=True,
            tickfont=dict(color="firebrick"),
            anchor="x",
        ),
        yaxis2=dict(
            title=dict(text="SPY Price", font=dict(color="royalblue")),
            overlaying="y",
            side="right",
            showgrid=False,
            showline=True,
            tickfont=dict(color="royalblue"),
            anchor="x",
        ),
        hovermode="x unified",
        template="plotly_white",
        legend=dict(x=0.01, y=0.99),
    )

    os.makedirs("output", exist_ok=True)
    fig.write_html("output/high_yield_vs_spy.html", include_plotlyjs="cdn")
    print("✅ Chart saved to: output/high_yield_vs_spy.html")

    fig.show(config={"scrollZoom": True})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=int, default=1, help="Years to display (default: 1)")
    args = parser.parse_args()

    fred_api_key = load_fred_api_key()
    effective_yield_series = get_spread_data(fred_api_key, args.years)
    treasury_series = get_treasury_rate_data(fred_api_key, args.years)
    credit_spread_series = get_credit_spread_data(fred_api_key, args.years)
    spy_series = get_spy_data(args.years)
    plot_combined(
        effective_yield_series,
        spy_series,
        treasury_series=treasury_series,
        credit_spread_series=credit_spread_series,
    )


if __name__ == "__main__":
    main()
