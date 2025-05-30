import argparse
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
        raise RuntimeError("FRED API key is missing! Save it to ./credentials/credential_fred_api.txt.")


def get_yield_data(fred: Fred, series_id: str, start_date: datetime) -> pd.Series:
    return fred.get_series(series_id, observation_start=start_date).rename(series_id)


def get_yield_curve_gap(fred_api_key: str, years: int) -> pd.DataFrame:
    fred = Fred(api_key=fred_api_key)
    start_date = datetime.today() - timedelta(days=365 * years)

    data_10y = get_yield_data(fred, "GS10", start_date)
    data_2y = get_yield_data(fred, "GS2", start_date)

    df = pd.concat([data_10y, data_2y], axis=1).dropna()
    df.columns = ["10Y", "2Y"]
    df["Spread"] = df["10Y"] - df["2Y"]
    return df


def get_spy_data(start_date: datetime) -> pd.Series:
    spy = yf.download("SPY", start=start_date.strftime("%Y-%m-%d"), progress=False)

    if isinstance(spy.columns, pd.MultiIndex):
        if ('Close', 'SPY') not in spy.columns:
            raise RuntimeError("Expected ('Close', 'SPY') column not found.")
        return spy[('Close', 'SPY')].rename("SPY")
    else:
        if "Adj Close" in spy.columns:
            return spy["Adj Close"].rename("SPY")
        elif "Close" in spy.columns:
            return spy["Close"].rename("SPY")
        else:
            raise RuntimeError("No suitable SPY price column found.")


def align_to_business_days(df: pd.DataFrame) -> pd.DataFrame:
    business_days = pd.date_range(df.index.min(), df.index.max(), freq="B")
    return df.reindex(business_days).ffill()


def plot_yield_and_spy(df: pd.DataFrame):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df.index, y=df["10Y"],
        mode="lines", name="10Y Yield", line=dict(width=1.5, color="blue"), yaxis="y1"
    ))

    fig.add_trace(go.Scatter(
        x=df.index, y=df["2Y"],
        mode="lines", name="2Y Yield", line=dict(width=1.5, color="orange"), yaxis="y1"
    ))

    fig.add_trace(go.Scatter(
        x=df.index, y=df["Spread"],
        mode="lines", name="10Y - 2Y Spread", line=dict(width=2, dash="dot", color="green"), yaxis="y1"
    ))

    fig.add_trace(go.Scatter(
        x=df.index, y=df["SPY"],
        mode="lines", name="SPY (Close)", line=dict(width=2, color="black"), yaxis="y2"
    ))

    fig.update_layout(
        title="Daily Yield Curve (10Y, 2Y, Spread) vs SPY",
        xaxis=dict(
            title="Date",
            rangeslider=dict(visible=True),
            type="date"
        ),
        yaxis=dict(
            title="Yield & Spread (%)",
            side="left"
        ),
        yaxis2=dict(
            title="SPY (Close)",
            overlaying="y",
            side="right"
        ),
        hovermode="x unified",
        dragmode="pan",
        legend=dict(x=0.01, y=0.99)
    )

    # ✅ Enable scroll wheel zoom
    fig.show(config=dict(scrollZoom=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=int, default=5, help="Number of years to display")
    args = parser.parse_args()

    fred_api_key = load_fred_api_key()
    df_spread = get_yield_curve_gap(fred_api_key, args.years)

    start_date = df_spread.index.min()
    df_spy = get_spy_data(start_date)

    df = pd.concat([df_spread, df_spy], axis=1)
    df = align_to_business_days(df).dropna()

    print(f"DEBUG: Date range: {df.index.min()} to {df.index.max()}")
    print(f"DEBUG: Spread range: {df['Spread'].min():.2f} to {df['Spread'].max():.2f}")
    print(f"DEBUG: 10Y range: {df['10Y'].min():.2f} to {df['10Y'].max():.2f}")
    print(f"DEBUG: 2Y range: {df['2Y'].min():.2f} to {df['2Y'].max():.2f}")
    print(f"DEBUG: SPY range: {df['SPY'].min():.2f} to {df['SPY'].max():.2f}")

    plot_yield_and_spy(df)


if __name__ == "__main__":
    main()

