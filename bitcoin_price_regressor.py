"""
Bitcoin Price Prediction and Analysis

As of March 3, 2025, the regression formula was:
30191.72 + -572.18*Dollar Index + -4179.16*2-Year Treasury Yield + 27825.68*S&P 500 + -4573.38*Log Net Liquidity

Ranking of impacts (the upper the highest):
S&P 500	+27825.68	🚀 Strong Positive	Higher S&P 500 → Higher Bitcoin
Log Net Liquidity	-4573.38	🔥 Strong Negative	Lower Liquidity → Lower Bitcoin
2-Year Treasury Yield	-4179.16	🔻 Strong Negative	Higher Yields → Lower Bitcoin
Dollar Index	-572.18	📉 Moderate Negative	Stronger USD → Lower Bitcoin

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
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from fredapi import Fred

# ✅ Configurable Percentage Gap for Buy/Sell Signals
PERCENT_GAP_THRESHOLD = 0.15  # Default 15%, adjust as needed

# ✅ Configurable period to investigate and to regress the bitcoin price
YEAR_WINDOW = 7

# Load FRED API key
# Try to load API key from environment variable
fred_api_key = os.getenv("FRED_API_KEY")

# If not found, fall back to reading from a file (for local execution)
if not fred_api_key:
    try:
        with open("./credentials/credential_fred_api.txt", "r") as file:
            fred_api_key = file.read().strip()
    except FileNotFoundError:
        raise RuntimeError("FRED API key is missing! Set it as an environment variable or store it in ./credentials/credential_fred_api.txt.")

# Debugging: Ensure API key is set (Optional: Remove in production)
if not fred_api_key:
    raise RuntimeError("FRED API key could not be loaded from either environment variable or file!")

print("API key loaded successfully.")  # For debugging purposes

#with open("./credentials/credential_fred_api.txt", "r") as file:
#    fred_api_key = file.read().strip()

# Initialize FRED API
fred = Fred(api_key=fred_api_key)

# Define the time range
end_date = datetime.now()
start_date = end_date - timedelta(days=YEAR_WINDOW * 365)

# Define ticker symbols for explanatory variables
tickers = {
    'BTC-USD': 'Bitcoin',
    'DX-Y.NYB': 'Dollar Index',
    'GLD': 'Gold Price',
    '^GSPC': 'S&P 500',
    '^VIX': 'VIX Index'
}

# Download data from Yahoo Finance
data = {}
for ticker, name in tickers.items():
    try:
        df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)
        data[name] = df['Adj Close'].squeeze() if 'Adj Close' in df.columns else df['Close'].squeeze()
    except Exception as e:
        print(f"Error downloading {name} data: {e}")

# Get 2-Year US Treasury Yield data from FRED
try:
    treasury_yield = fred.get_series('DGS2', start_date, end_date)
    data['2-Year Treasury Yield'] = treasury_yield
except Exception as e:
    print(f"Error downloading 2-Year Treasury Yield data from FRED: {e}")

# Get Net Liquidity Components
try:
    fed_balance_sheet = fred.get_series('WALCL', start_date, end_date)
    reverse_repo = fred.get_series('RRPONTSYD', start_date, end_date)
    treasury_general_account = fred.get_series('WTREGEN', start_date, end_date)

    # Compute Net Liquidity
    data['Net Liquidity'] = fed_balance_sheet - reverse_repo - treasury_general_account

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Fix missing/zero Net Liquidity values
    df['Net Liquidity'] = df['Net Liquidity'].replace(0, np.nan)  # Avoid log(0) issue
    df['Net Liquidity'] = df['Net Liquidity'].ffill()  # Forward-fill missing values
    df['Log Net Liquidity'] = np.log(df['Net Liquidity'])  # Compute log after ffill

except Exception as e:
    print(f"Error downloading Net Liquidity data: {e}")

# Print column names for debugging
print("\nColumns in dataframe:", df.columns.tolist())

# Define expected features (Using Log Net Liquidity instead of raw Net Liquidity)
expected_features = ['Dollar Index', '2-Year Treasury Yield', 'S&P 500', 'Log Net Liquidity']
missing_features = [feature for feature in expected_features if feature not in df.columns]

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
    regression_formula = f"{model.params.iloc[0]:.2f} + " + " + ".join(
        [f"{coef:.2f}*{name}" for name, coef in zip(model.params.index[1:], model.params.iloc[1:])]
    )
    print("\n📌 Regression Formula:\n" + regression_formula)

    # Compute VIF to check multicollinearity
    vif_data = pd.DataFrame()
    vif_data['Variable'] = X.columns
    vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
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
    <b>Regression Formula:</b><br>{regression_formula}<br>
    <b>R-Squared:</b> {model.rsquared:.3f}<br>
    <b>Variance Inflation Factor (VIF):</b><br>{"<br>".join([f"{var}: {vif:.2f}" for var, vif in zip(vif_data['Variable'], vif_data['VIF'])])}
    """

    # Create the figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Bitcoin'], mode='lines', name='Actual Bitcoin Price'))
    fig.add_trace(go.Scatter(x=df.index, y=df['Predicted Bitcoin'], mode='lines', name='Predicted Bitcoin Price'))
    fig.add_trace(go.Scatter(x=df.index, y=df['100-Day MA'], mode='lines', line=dict(dash='dash', color='darkblue', width=2), name='100-Day Moving Average'))
    fig.add_trace(go.Scatter(x=df.index, y=df['10-Day MA'], mode='lines', line=dict(dash='dot', color='skyblue', width=2), name='10-Day Moving Average'))
    fig.add_trace(go.Scatter(x=df.index, y=df['Log Net Liquidity'], mode='lines', line=dict(color='purple', width=2), name='Log Net Liquidity', yaxis='y2'))
    fig.add_trace(go.Scatter(x=df[df['Buy Signal']].index, y=df[df['Buy Signal']]['Bitcoin'], mode='markers', marker=dict(size=10, color='blue', symbol='triangle-up'), name='Buy Signal (▲)'))
    fig.add_trace(go.Scatter(x=df[df['Sell Signal']].index, y=df[df['Sell Signal']]['Bitcoin'], mode='markers', marker=dict(size=10, color='red', symbol='triangle-down'), name='Sell Signal (▼)'))
    fig.add_annotation(text=stats_text, xref='paper', yref='paper', x=0.02, y=0.98, showarrow=False, align="left", font=dict(size=12))

    fig.update_layout(title='Predicted vs Actual Bitcoin Prices', xaxis=dict(rangeslider=dict(visible=True), type='date'), yaxis=dict(title='Bitcoin Price (USD)'), yaxis2=dict(title='Log Net Liquidity', overlaying='y', side='right'))

def main():
    # Add argument parser
    parser = argparse.ArgumentParser(description="Bitcoin Price Prediction and Analysis")
    parser.add_argument("--output", type=str, help="Output HTML file to save the graph")
    args = parser.parse_args()

    # Check if an output file is specified
    if args.output:
        fig.write_html(args.output)
        print(f"Graph saved to {args.output}")
    else:
        fig.show()

if __name__ == "__main__":
    main()
