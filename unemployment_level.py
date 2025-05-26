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


def get_unemployment_data(fred_api_key: str, years: int):
    fred = Fred(api_key=fred_api_key)
    start_date = datetime.today() - timedelta(days=365 * years)
    data = fred.get_series("UNEMPLOY", observation_start=start_date)
    data.index = data.index.date
    data = data / 1000  # Convert to millions
    return data


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


def plot_combined(unemploy_series_raw, spy_series_raw):
    x_dates = unemploy_series_raw.index

    # Reindex both series to match the same date index
    spy_series = spy_series_raw.reindex(x_dates).ffill()
    unemploy_series = unemploy_series_raw.reindex(x_dates).ffill()

    # MA10s
    unemploy_ma10 = unemploy_series.rolling(window=10).mean()
    spy_ma10 = spy_series.rolling(window=10).mean()

    print("DEBUG: Unemployment range (M):", unemploy_series.min(), "to", unemploy_series.max())
    print("DEBUG: SPY range:", spy_series.min(), "to", spy_series.max())
    print("DEBUG: Series length:", len(spy_series))

    fig = go.Figure()

    # Unemployment level
    fig.add_trace(go.Scatter(
        x=x_dates,
        y=unemploy_series.values,
        name="Unemployment Level (M)",
        yaxis="y1",
        line=dict(color="darkgreen", width=2)
    ))

    # Unemployment MA10
    fig.add_trace(go.Scatter(
        x=x_dates,
        y=unemploy_ma10.values,
        name="Unemployment MA10",
        yaxis="y1",
        line=dict(color="darkgreen", width=1.5, dash="dot")
    ))

    # SPY
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

    fig.update_layout(
        title="Unemployment vs. SPY (with MA10)",
        dragmode="pan",
        xaxis=dict(
            title="Date",
            rangeslider=dict(visible=True),
            type="date"
        ),
        yaxis=dict(
            title=dict(text="Unemployment (Millions)", font=dict(color="darkgreen")),
            side="left",
            showgrid=True,
            showline=True,
            tickfont=dict(color="darkgreen"),
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
    fig.write_html("output/unemployment_vs_spy.html", include_plotlyjs="cdn")
    print("✅ Chart saved to: output/unemployment_vs_spy.html")

    fig.show(config={"scrollZoom": True})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=int, default=50, help="Years to display (default: 50)")
    args = parser.parse_args()

    fred_api_key = load_fred_api_key()
    unemploy_series = get_unemployment_data(fred_api_key, args.years)
    spy_series = get_spy_data(args.years)
    plot_combined(unemploy_series, spy_series)


if __name__ == "__main__":
    main()

