'''
gold_price_regressor.py

We'd like to regress gold price using the factors known to affect its price.
The factors we selected are:
  - 2-year treasury yield
  - dollar index
  - s&p500
  - vix index

After we got the coeff of each  and we combine together to regress gold price.
We also render point to buy when the actual price is less than
{price_gap} percentage of the predicted price. In a similar way, we show
point to sell when the actuall price is more than {price_gap} percentage of the
predicted price.
'''

import argparse
import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import yfinance as yf
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Argument parser for optional output filename
parser = argparse.ArgumentParser(description="Regress gold price using selected economic factors.")
parser.add_argument("--output", type=str, default=None, help="Output file path for saving the HTML graph.")
args = parser.parse_args()

years_window=3 # years span
price_gap=0.05 # for gap detector, 5%

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

# Define the time period (last {years_window} years)
#end_date = datetime.today().strftime('%Y-%m-%d')
end_date = (datetime.today() + timedelta(days=1)).strftime('%Y-%m-%d')

print(f"end_date: {end_date}")
start_date = (datetime.today() - timedelta(days=years_window*365)).strftime('%Y-%m-%d')
print(f"start_date: {start_date}")

# Define tickers for yfinance
tickers = {
    "gold_price": "GC=F",  # Gold Futures
    "sp500": "^GSPC",      # S&P 500 Index
    "dollar_index": "DX-Y.NYB",  # Dollar Index
    "vix": "^VIX"  # Market Volatility Index (VIX)
}

# Fetch data from yfinance
data = {}
for key, ticker in tickers.items():
    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)[["Close"]]
    if df is not None and not df.empty:
        data[key] = df["Close"].squeeze()

# Convert dictionary to DataFrame
df_yf = pd.DataFrame(data)
print('----df_yf:')
print(df_yf.tail())

# Fetch latest available S&P 500 price
latest_sp500 = yf.download("^GSPC", period="1d", interval="1h")
print('latest_sp500:')
print(latest_sp500)
if not latest_sp500.empty:
  # Convert latest_sp500 index to datetime without timezone
  latest_sp500.index = latest_sp500.index.tz_localize(None)

  # Extract last available trading date and price
  last_trade_date = latest_sp500.index[-1].date()  # Convert to date only
  last_price = latest_sp500["Close"].iloc[-1]

  # Ensure df_yf index is datetime format
  df_yf.index = pd.to_datetime(df_yf.index).date  # Convert to date format

  # Update or append latest S&P 500 price
  if last_trade_date in df_yf.index:
    df_yf.at[last_trade_date, "sp500"] = last_price  # Update existing row
  else:
    new_row = pd.DataFrame({"gold_price": [np.nan], "sp500": [last_price],
                            "dollar_index": [np.nan], "vix": [np.nan]},
                            index=[last_trade_date])
    df_yf = pd.concat([df_yf, new_row]).sort_index()  # Append and sort

# if sp500 is still missing, forward-fill
df_yf["sp500"].fillna(method="ffill", inplace=True)
print('----df_yf with latest update:')
print(df_yf.tail())

# Fetch 2-Year Treasury Yield from FRED (if API key available)
if fred_api_key:
    from fredapi import Fred
    fred = Fred(api_key=fred_api_key)
    treasury_2y_series = fred.get_series("DGS2", start_date, end_date)
    df_fred = pd.DataFrame(treasury_2y_series, columns=["treasury_2y"])
    df_fred.index = pd.to_datetime(df_fred.index)
    df_fred = df_fred.ffill()
else:
    df_fred = None
print('----df_fred:')
print(df_fred.tail())

# Merge all datasets
df = df_yf.merge(df_fred, left_index=True, right_index=True, how="left")
df = df.ffill()
print('----df after merging all datasets:')
print(df)

# Define independent variables
X = df[["dollar_index", "treasury_2y", "sp500", "vix"]]
y = df["gold_price"]

# Add constant for regression
X = sm.add_constant(X)

# Fit the regression model
model = sm.OLS(y, X).fit()

# Get regression coefficients
coeffs = model.params
r_squared = model.rsquared
print("----coeffs:")
print(coeffs)
print("----r_squared:")
print(r_squared)

# Compute predicted gold prices
df["Predicted Gold Price"] = model.predict(X)

# Compute VIF values
vif_data = pd.DataFrame()
vif_data["Variable"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print("----vif_data:")
print(vif_data)

# Create separate text annotations
regression_formula = f"Gold Price = {coeffs['const']:.2f} + ({coeffs['dollar_index']:.2f} * Dollar Index) + ({coeffs['treasury_2y']:.2f} * Treasury 2Y) + ({coeffs['sp500']:.2f} * S&P 500) + ({coeffs['vix']:.2f} * VIX)"
r_squared_text = f"R-squared: {r_squared:.4f}"
vif_text = "VIF Values:\n" + "\n".join([f"{var}: {vif:.2f}" for var, vif in zip(vif_data["Variable"], vif_data["VIF"])])

# Identify buy and sell signals
df["Buy Signal"] = df["Predicted Gold Price"] > df["gold_price"] * (1+price_gap)
df["Sell Signal"] = df["Predicted Gold Price"] < df["gold_price"] * (1-price_gap)

# Create the Plotly figure
fig = go.Figure()

# Add actual gold price trace
fig.add_trace(go.Scatter(
    x=df.index,
    y=df["gold_price"],
    mode="lines",
    name="Actual Gold Price",
    line=dict(color="blue")
))

# Add predicted gold price trace
fig.add_trace(go.Scatter(
    x=df.index,
    y=df["Predicted Gold Price"],
    mode="lines",
    name="Predicted Gold Price",
    line=dict(color="orange", dash="solid")
))

# Add buy signal markers
fig.add_trace(go.Scatter(
    x=df.index[df["Buy Signal"]],
    y=df["gold_price"][df["Buy Signal"]],
    mode="markers",
    marker=dict(symbol="triangle-up", size=10, color="green"),
    name=f"Buy Signal Based on Price Gap {price_gap}"
))

# Add sell signal markers
fig.add_trace(go.Scatter(
    x=df.index[df["Sell Signal"]],
    y=df["gold_price"][df["Sell Signal"]],
    mode="markers",
    marker=dict(symbol="triangle-down", size=10, color="red"),
    name=f"Sell Signal Based on Price Gap {price_gap}"
))

# Add regression formula annotation
fig.add_annotation(
    x=df.index[len(df) // 2],
    y=df["gold_price"].max(),
    text=regression_formula,
    showarrow=False,
    font=dict(size=12, color="black"),
    align="left",
    bordercolor="black",
    borderwidth=1,
    bgcolor="white"
)

# Add R-squared annotation
fig.add_annotation(
    x=df.index[len(df) // 2],
    y=df["gold_price"].max() * 0.95,
    text=r_squared_text,
    showarrow=False,
    font=dict(size=12, color="black"),
    align="left",
    bordercolor="black",
    borderwidth=1,
    bgcolor="white"
)

# Add VIF annotation
fig.add_annotation(
    x=df.index[len(df) // 2],
    y=df["gold_price"].max() * 0.90,
    text=vif_text.replace("\n", "<br>"),
    showarrow=False,
    font=dict(size=12, color="black"),
    align="left",
    bordercolor="black",
    borderwidth=1,
    bgcolor="white"
)

# Add slider for zooming
fig.update_layout(
    title=f"Actual vs Predicted Gold Price (Last {years_window} Years) with Buy/Sell Signals",
    xaxis_title="Date",
    yaxis_title="Gold Price (USD)",
    legend_title="Legend",
    template="plotly_white",
    xaxis=dict(
        rangeslider=dict(visible=True),
        type="date"
    )
)

# Save the plot to an HTML file if output is provided, otherwise show it in the browser
if args.output:
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)  # Ensure directory exists
    fig.write_html(args.output)
    print(f"Graph saved to {args.output}")
else:
    fig.show()


