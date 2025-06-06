import argparse
import os
from datetime import datetime, timedelta
import pandas as pd
from fredapi import Fred
import yfinance as yf
import plotly.graph_objects as go


def load_fred_api_key():
    try:
        with open("./credentials/credential_fred_api.txt", "r") as file:
            return file.read().strip()
    except FileNotFoundError:
        raise RuntimeError(
            "FRED API key is missing! Store it in ./credentials/credential_fred_api.txt."
        )


def get_spread_data(fred_api_key: str, years: int):
    fred = Fred(api_key=fred_api_key)
    start_date = datetime.today() - timedelta(days=365 * years)
    series = fred.get_series("BAMLH0A0HYM2", observation_start=start_date)
    series.index = series.index.date
    return series


def get_spy_data(years: int):
    end = datetime.today()
    start = end - timedelta(days=365 * years)
    spy_df = yf.download(tickers="SPY", start=start, end=end, progress=False, auto_adjust=True)

    if isinstance(spy_df.columns, pd.MultiIndex):
        close_series = spy_df[("Close", "SPY")]
    else:
        close_series = spy_df["Close"]

    close_series.name = "SPY"
    close_series.index = close_series.index.date
    return close_series


def plot_combined(spread_series, spy_series_raw):
    x_dates = spread_series.index
    spread_series = spread_series.reindex(x_dates).ffill()
    spy_series = spy_series_raw.reindex(x_dates).ffill()

    # Compute 10-day moving averages
    oas_ma10 = spread_series.rolling(window=10).mean()
    spy_ma10 = spy_series.rolling(window=10).mean()

    # Debug info
    print("DEBUG: OAS range:", spread_series.min(), "to", spread_series.max())
    print("DEBUG: SPY range:", spy_series.min(), "to", spy_series.max())
    print("DEBUG: Date range:", x_dates.min(), "to", x_dates.max())
    print("DEBUG: Series length:", len(spread_series), len(spy_series))

    fig = go.Figure()

    # OAS line
    fig.add_trace(go.Scatter(
        x=x_dates,
        y=spread_series.values,
        name="High-Yield OAS (%)",
        yaxis="y1",
        line=dict(color="firebrick", width=2)
    ))

    # OAS MA10
    fig.add_trace(go.Scatter(
        x=x_dates,
        y=oas_ma10.values,
        name="OAS MA10",
        yaxis="y1",
        line=dict(color="firebrick", width=1.5, dash="dot")
    ))

    # SPY line
    fig.add_trace(go.Scatter(
        x=x_dates,
        y=spy_series.values,
        name="SPY Price",
        yaxis="y2",
        line=dict(color="royalblue", width=2)
    ))

    # SPY MA10
    fig.add_trace(go.Scatter(
        x=x_dates,
        y=spy_ma10.values,
        name="SPY MA10",
        yaxis="y2",
        line=dict(color="royalblue", width=1.5, dash="dot")
    ))

    # Reference lines at 4% and 6%
    for level in [4, 6]:
        fig.add_hline(
            y=level,
            line=dict(color="gray", dash="dot"),
            annotation_text=f"{level}%",
            annotation_position="top right"
        )

    # Layout
    fig.update_layout(
        title="High-Yield Spread vs. SPY (Overlapped, with MA10)",
        dragmode="pan",
        xaxis=dict(
            title="Date",
            rangeslider=dict(visible=True),
            type="date"
        ),
        yaxis=dict(
            title=dict(text="OAS (%)", font=dict(color="firebrick")),
            side="left",
            showgrid=True,
            showline=True,
            tickfont=dict(color="firebrick"),
            anchor="x"
        ),
        yaxis2=dict(
            title=dict(text="SPY Price", font=dict(color="royalblue")),
            overlaying="y",
            side="right",
            showgrid=False,
            showline=True,
            tickfont=dict(color="royalblue"),
            anchor="x"
        ),
        hovermode="x unified",
        template="plotly_white",
        legend=dict(x=0.01, y=0.99)
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
    spread_series = get_spread_data(fred_api_key, args.years)
    spy_series = get_spy_data(args.years)
    plot_combined(spread_series, spy_series)


if __name__ == "__main__":
    main()

