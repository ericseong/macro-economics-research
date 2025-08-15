"""
Bitcoin Price Prediction and Analysis

As of March 3, 2025, the regression formula was:
30191.72 + -572.18*Dollar Index + -4179.16*2-Year Treasury Yield + 27825.68*S&P 500 + -4573.38*Log Net Liquidity

Ranking of impacts:
S&P 500:
+1% S&P 500 ≈ +0.92% BTC

2-Year Treasury Yield:
+10 bps (0.10 percentage points) in the 2-year yield ≈ −0.68% BTC

Comment:
Lower liquidity -> higher bitcoin price? Since 2023, lower liquidity corresponds
to higher bitcoin price. This is strange and not intuitive.

Note:
- Downloads Bitcoin price data and key macroeconomic indicators.
- Builds an OLS regression model to predict Bitcoin prices.
- Adds Buy (▲) and Sell (▼) signals based on a configurable percentage gap.
- Displays actual and predicted prices with moving averages.
- Includes Net Liquidity as a secondary Y-axis.
- Provides a statistical summary (R², VIF, and regression formula) in the chart.
- Allows toggling Buy/Sell signals in the legend.
"""

import os
import argparse
import yfinance as yf
from curl_cffi import requests
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from fredapi import Fred

# Add argument parser
parser = argparse.ArgumentParser(
    description="Bitcoin Price Prediction and Analysis")
parser.add_argument("--output",
                    type=str,
                    help="Output HTML file to save the graph")
parser.add_argument("--years",
                    type=int,
                    default=7,
                    help="Number of years for data retrieval.")
args = parser.parse_args()

# ✅ Configurable Percentage Gap for Buy/Sell Signals
PERCENT_GAP_THRESHOLD = 0.15  # Default 15%, adjust as needed

# ✅ we retry downloading after changing the start_date.
# This backup has been added to avoid download failure, specifically for VIX,
# for a specific start_date with yfinance
MAX_ATTEMPTS = 5

# Load FRED API key
# Try to load API key from environment variable
fred_api_key = os.getenv("FRED_API_KEY")

# If not found, fall back to reading from a file (for local execution)
if not fred_api_key:
    try:
        with open("./credentials/credential_fred_api.txt", "r") as file:
            fred_api_key = file.read().strip()
    except FileNotFoundError:
        raise RuntimeError(
            "FRED API key is missing! Set it as an environment variable or store it in ./credentials/credential_fred_api.txt."
        )

# Debugging: Ensure API key is set (Optional: Remove in production)
if not fred_api_key:
    raise RuntimeError(
        "FRED API key could not be loaded from either environment variable or file!"
    )

print("API key loaded successfully.")  # For debugging purposes

#with open("./credentials/credential_fred_api.txt", "r") as file:
#    fred_api_key = file.read().strip()

# Initialize FRED API
fred = Fred(api_key=fred_api_key)

# Define the time range
end_date = datetime.now()
start_date = end_date - timedelta(days=args.years * 365)

# Define ticker symbols for explanatory variables
tickers = {
    'BTC-USD': 'Bitcoin',
    'DX-Y.NYB': 'Dollar Index',
    'GLD': 'Gold Price',
    '^GSPC': 'S&P 500',
    '^VIX': 'VIX Index'
}


def debug_print(text: str, data: pd.Series, freq: str):
    print(text)
    data.index = pd.to_datetime(data.index)
    latest_date = data.index.max()
    if freq == 'year':
        sampled_data = data.groupby(data.index.year).apply(lambda x: x.iloc[
            np.argmin(np.abs((x.index - latest_date).total_seconds()))])
    elif freq == 'month':
        sampled_data = data.groupby(
            data.index.to_period('M')).apply(lambda x: x.iloc[np.argmin(
                np.abs((x.index - latest_date).total_seconds()))])
    else:
        print("Invalid frequency. Use 'year' or 'month'.")
    sampled_data = sampled_data.sort_index(ascending=False)
    print(sampled_data)


# Download data from Yahoo Finance with some retry trials
data = {}
for ticker, name in tickers.items():
    attempt = 0
    _start_date = start_date
    while attempt < MAX_ATTEMPTS:
        try:
            session = requests.Session(impersonate="chrome")
            df = yf.download(ticker,
                             start=_start_date,
                             end=end_date,
                             auto_adjust=False,
                             session=session)
            if df.empty:
                print(
                    f"No data retrieved for {name} ({ticker} with start_date {_start_date.date()}"
                )
                raise ValueError("Empty DataFrame returned")
            data[name] = df['Adj Close'].squeeze(
            ) if 'Adj Close' in df.columns else df['Close'].squeeze()
            print(
                f"In {attempt}th trial, downloaded {name}: {len(data[name])} rows (start_date: {_start_date.date()})"
            )
            break
        except Exception as e:
            print(f"Error downloading {name} data: {e}")
            attempt += 1
            if attempt < MAX_ATTEMPTS:
                _start_date = _start_date - timedelta(days=1)
            else:
                print(
                    f"Failed to download {name} ({ticker}) after {MAX_ATTEMPTS} attempts."
                )
                data[name] = pd.Series(dtype=float)

# Get 2-Year US Treasury Yield data from FRED
try:
    treasury_yield = fred.get_series('DGS2', start_date, end_date)
    data['2-Year Treasury Yield'] = treasury_yield
except Exception as e:
    print(f"Error downloading 2-Year Treasury Yield data from FRED: {e}")

# Get Net Liquidity Components
try:
    fed_balance_sheet = fred.get_series('WALCL', start_date, end_date)
    debug_print('fed_balance_sheet (M$)', fed_balance_sheet, 'year')
    reverse_repo = fred.get_series('RRPONTSYD', start_date, end_date)
    debug_print('reverse_repo (M$)', reverse_repo, 'year')
    treasury_general_account = fred.get_series('WTREGEN', start_date, end_date)
    debug_print('treasury_general_account (M$)', treasury_general_account,
                'year')

    # Compute Net Liquidity
    data[
        'Net Liquidity'] = fed_balance_sheet - reverse_repo - treasury_general_account

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Fix missing/zero Net Liquidity values
    df['Net Liquidity'] = df['Net Liquidity'].replace(
        0, np.nan)  # Avoid log(0) issue
    df['Net Liquidity'] = df['Net Liquidity'].ffill(
    )  # Forward-fill missing values
    df['Log Net Liquidity'] = np.log(
        df['Net Liquidity'])  # Compute log after ffill

except Exception as e:
    print(f"Error downloading Net Liquidity data: {e}")

# Print column names for debugging
print("\nColumns in dataframe:", df.columns.tolist())

# Define expected features (Using Log Net Liquidity instead of raw Net Liquidity)
expected_features = [
    'Dollar Index', '2-Year Treasury Yield', 'S&P 500', 'Log Net Liquidity'
]
missing_features = [
    feature for feature in expected_features if feature not in df.columns
]

if missing_features:
    print(f"Warning: The following features are missing: {missing_features}")

# Ensure the index stays daily-based
df = df.asfreq('D', method='ffill')  # Maintain daily granularity

# Forward-fill missing values
df = df.ffill()

# Drop rows where Bitcoin or other essential data is missing
df.dropna(inplace=True)

# **Standardize features (Z-score scaling)**
scaler = StandardScaler()
df[expected_features] = scaler.fit_transform(df[expected_features])

# Compute moving averages
df['200-Day MA'] = df['Bitcoin'].rolling(window=200).mean()
df['100-Day MA'] = df['Bitcoin'].rolling(window=100).mean()
df['10-Day MA'] = df['Bitcoin'].rolling(window=10).mean()

# Perform regression if all expected features exist
if all(feature in df.columns for feature in expected_features):
    X = df[expected_features]
    y = df['Bitcoin']
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()

    print(model.summary())

    # ✅ Print Regression Formula in Console
    regression_formula = f"{model.params.iloc[0]:.2f} + " + " + ".join([
        f"{coef:.2f}*{name}"
        for name, coef in zip(model.params.index[1:], model.params.iloc[1:])
    ])
    print("\n📌 Regression Formula:\n" + regression_formula)

    # Compute VIF to check multicollinearity
    vif_data = pd.DataFrame()
    vif_data['Variable'] = X.columns
    vif_data['VIF'] = [
        variance_inflation_factor(X.values, i) for i in range(X.shape[1])
    ]
    print("\nVariance Inflation Factor (VIF):")
    print(vif_data)

    # Predict Bitcoin prices
    df['Predicted Bitcoin'] = model.predict(X)

    # Identify buy/sell signals based on configurable gap
    df['Error'] = (df['Predicted Bitcoin'] - df['Bitcoin']) / df['Bitcoin']
    df['Buy Signal'] = df['Error'] >= PERCENT_GAP_THRESHOLD
    df['Sell Signal'] = df['Error'] <= -PERCENT_GAP_THRESHOLD

    # ✅ Preserve **Exact** Stat Summary Formatting
    stats_text = f"""
    <br>Regression Formula:{regression_formula}
    <br>R-Squared: {model.rsquared:.3f}
    <br>Variance Inflation Factor (VIF):{"<br>".join([f"{var}: {vif:.2f}" for var, vif in zip(vif_data['Variable'], vif_data['VIF'])])}
    """

    # Create the figure
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=df.index,
                   y=df['Bitcoin'],
                   mode='lines',
                   line=dict(color='orange', width=2),
                   name='Actual Bitcoin Price'))
    fig.add_trace(
        go.Scatter(x=df.index,
                   y=df['Predicted Bitcoin'],
                   mode='lines',
                   line=dict(color='skyblue', width=2),
                   name='Predicted Bitcoin Price'))
    fig.add_trace(
        go.Scatter(x=df.index,
                   y=df['200-Day MA'],
                   mode='lines',
                   line=dict(dash='dot', color='lightgray', width=3),
                   name='200-Day Moving Average'))
    fig.add_trace(
        go.Scatter(x=df.index,
                   y=df['100-Day MA'],
                   mode='lines',
                   line=dict(dash='dot', color='lightgray', width=2),
                   name='100-Day Moving Average'))
    fig.add_trace(
        go.Scatter(x=df.index,
                   y=df['10-Day MA'],
                   mode='lines',
                   line=dict(dash='dot', color='lightgray', width=1),
                   name='10-Day Moving Average'))
    fig.add_trace(
        go.Scatter(x=df.index,
                   y=df['Log Net Liquidity'],
                   mode='lines',
                   line=dict(color='darkgray', width=2),
                   name='Log Net Liquidity',
                   yaxis='y2'))
    fig.add_trace(
        go.Scatter(x=df[df['Buy Signal']].index,
                   y=df[df['Buy Signal']]['Bitcoin'],
                   mode='markers',
                   marker=dict(size=8, color='white', symbol='triangle-up'),
                   name='Buy Signal (▲)'))
    fig.add_trace(
        go.Scatter(x=df[df['Sell Signal']].index,
                   y=df[df['Sell Signal']]['Bitcoin'],
                   mode='markers',
                   marker=dict(size=8, color='yellow', symbol='triangle-down'),
                   name='Sell Signal (▼)'))
    fig.add_annotation(text=stats_text,
                       xref='paper',
                       yref='paper',
                       x=0.02,
                       y=0.98,
                       showarrow=False,
                       align="left",
                       font=dict(size=12))

    fig.update_layout(
        title=
        f'Predicted vs Actual Bitcoin Price for the recent {args.years} years with normalized factors',
        xaxis=dict(
            rangeslider=dict(visible=True),
            type='date',
            fixedrange=False  # allows zooming on x-axis
        ),
        yaxis=dict(
            title='Bitcoin Price (USD)',
            fixedrange=True  # prevents zooming on y-axis
        ),
        yaxis2=dict(
            title='Log Net Liquidity',
            overlaying='y',
            side='right',
            fixedrange=True  # Prevents zooming on secondary y-axis
        ),
        dragmode='pan',
        template="plotly_dark")


def main():
    # Check if an output file is specified
    if args.output:
        fig.write_html(args.output,
                       config={
                           'scrollZoom': True,
                           'modeBarButtonsToAdd': ['pan2d']
                       })
        print(f"Graph saved to {args.output}")
    else:
        fig.show(config={'scrollZoom': True, 'modeBarButtonsToAdd': ['pan2d']})


if __name__ == "__main__":
    main()
