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
from datetime import datetime, timedelta
import yfinance as yf
from statsmodels.stats.outliers_influence import variance_inflation_factor


# Argument parser for optional output filename
parser = argparse.ArgumentParser(
    description="Regress gold price using selected economic factors.")
parser.add_argument("--years",
                    type=int,
                    default=3,
                    help="Number of years for data retrieval.")
parser.add_argument("--output",
                    type=str,
                    default=None,
                    help="Output file path for saving the HTML graph.")
args = parser.parse_args()

years_window = args.years  # years span
price_gap = 0.05  # for gap detector, 5%
MAX_ATTEMPTS = 5  # max retry count for downloading the data frame

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

# Define the time period (last {years_window} years)
end_date = (datetime.today() + timedelta(days=1))

print(f"end_date: {end_date}")
#start_date = (datetime.today() - timedelta(days=years_window*365)).strftime('%Y-%m-%d')
start_date = (datetime.today() - timedelta(days=years_window * 365))
print(f"start_date: {start_date}")

# Define tickers for yfinance
tickers = {
    "gold_price": "GC=F",  # Gold Futures
    "sp500": "^GSPC",  # S&P 500 Index
    "dollar_index": "DX-Y.NYB",  # Dollar Index
    "vix": "^VIX"  # Market Volatility Index (VIX)
}

# Fetch data from yfinance
data = {}
for key, ticker in tickers.items():
    attempt = 0
    _start_date = start_date
    while attempt < MAX_ATTEMPTS:
        try:
            df = yf.download(ticker,
                             start=_start_date,
                             end=end_date,
                             auto_adjust=False)[["Close"]]
            if df is not None and not df.empty:
                data[key] = df["Close"].squeeze()
                break  # Success
            else:
                print(
                    f"No data retrieved for {key} ({ticker}) with start_date {_start_date.date()}"
                )
                raise ValueError("Empty DataFrame returned")
        except Exception as e:
            print(
                f"Error downloading {key} ({ticker}) with start_date {_start_date.date()}: {e}"
            )
            attempt += 1
            if attempt < MAX_ATTEMPTS:
                _start_date = _start_date - timedelta(
                    days=1)  # Fall back one day
                print(
                    f"Retrying with start_date {_start_date.date()} (attempt {attempt + 1}/{MAX_ATTEMPTS})"
                )
            else:
                print(
                    f"Failed to download {key} ({ticker}) after {MAX_ATTEMPTS} attempts."
                )
                data[key] = pd.Series(dtype=float)  # Empty series as fallback

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
        #df_yf.at[last_trade_date, "sp500"] = last_price  # Update existing row
        df_yf.at[last_trade_date, "sp500"] = float(last_price.iloc[0])
    else:
        new_row = pd.DataFrame(
            {
                "gold_price": [np.nan],
                "sp500": [last_price],
                "dollar_index": [np.nan],
                "vix": [np.nan]
            },
            index=[last_trade_date])
        df_yf = pd.concat([df_yf, new_row]).sort_index()  # Append and sort

# if sp500 is still missing, forward-fill
df_yf["sp500"] = df_yf["sp500"].ffill()
print('----df_yf with latest update:')
print(df_yf.tail())

start_date = start_date.strftime('%Y-%m-%d')
end_date = end_date.strftime('%Y-%m-%d')
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

# Make sure that there's no NaN value for the oldest row
while df.loc[df.index.min()].isna().any():
    oldest_date = df.index.min()  # Get the oldest date
    df = df.drop(index=oldest_date)  # Drop the row with the oldest date
print('----df after dropping the oldest row if NaN value exists:')
print(df)

# Define independent variables
X = df[["dollar_index", "treasury_2y", "sp500", "vix"]]
y = df["gold_price"]

X_scaled = (X - X.min()) / (X.max() - X.min())  # Scale to [0,1] range

X_scaled = sm.add_constant(X_scaled)  # Add constant term

# Fit model with scaled features
model = sm.OLS(y, X_scaled).fit()

# Extract coefficients
coeffs = model.params

print("----coeffs scaled----")
print(coeffs)

r_squared = model.rsquared
print("----r_squared:")
print(r_squared)

# Compute predicted gold prices
df["Predicted Gold Price"] = model.predict(X_scaled)

# Compute VIF values
'''
vif_data = pd.DataFrame()
vif_data["Variable"] = X.columns
vif_data["VIF"] = [
    variance_inflation_factor(X.values, i) for i in range(X.shape[1])
]
'''
# Calculate VIF without the constant term
X_vif = X_scaled.drop(columns=["const"],
                      errors="ignore")  # Remove constant for VIF calculation
vif_data = pd.DataFrame()
vif_data["Variable"] = X_vif.columns
vif_data["VIF"] = [
    variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])
]

print("----vif_data after removing constant:")
print(vif_data)

# Create separate text annotations
regression_formula = f"Gold Price = {coeffs['const']:.2f} + ({coeffs['dollar_index']:.2f} * Dollar Index) + ({coeffs['treasury_2y']:.2f} * Treasury 2Y) + ({coeffs['sp500']:.2f} * S&P 500) + ({coeffs['vix']:.2f} * VIX)"
r_squared_text = f"R-squared: {r_squared:.4f}"
vif_text = "VIF Values:\n" + "\n".join([
    f"{var}: {vif:.2f}"
    for var, vif in zip(vif_data["Variable"], vif_data["VIF"])
])

stats_text = f"""
<br>Regression Formula:{regression_formula}
<br>R-Squared: {model.rsquared:.3f}
<br>Variance Inflation Factor (VIF):{"<br>".join([f"{var}: {vif:.2f}" for var, vif in zip(vif_data['Variable'], vif_data['VIF'])])}
"""

# Identify buy and sell signals
df["Buy Signal"] = df["Predicted Gold Price"] > df["gold_price"] * (1 +
                                                                    price_gap)
df["Sell Signal"] = df["Predicted Gold Price"] < df["gold_price"] * (1 -
                                                                     price_gap)

# Create the Plotly figure
fig = go.Figure()

# Add actual gold price trace
fig.add_trace(
    go.Scatter(x=df.index,
               y=df["gold_price"],
               mode="lines",
               name="Actual Gold Price",
               line=dict(color="gold")))

# Add predicted gold price trace
fig.add_trace(
    go.Scatter(x=df.index,
               y=df["Predicted Gold Price"],
               mode="lines",
               name="Predicted Gold Price",
               line=dict(color="skyblue", dash="solid")))

# Add buy signal markers
fig.add_trace(
    go.Scatter(x=df.index[df["Buy Signal"]],
               y=df["gold_price"][df["Buy Signal"]],
               mode="markers",
               marker=dict(symbol="triangle-up", size=10, color="white"),
               name=f"Buy Signal Based on Price Gap {price_gap}"))

# Add sell signal markers
fig.add_trace(
    go.Scatter(x=df.index[df["Sell Signal"]],
               y=df["gold_price"][df["Sell Signal"]],
               mode="markers",
               marker=dict(symbol="triangle-down", size=10, color="yellow"),
               name=f"Sell Signal Based on Price Gap {price_gap}"))

# Add some informational text.
fig.add_annotation(text=stats_text,
                   xref='paper',
                   yref='paper',
                   x=0.02,
                   y=0.98,
                   showarrow=False,
                   align="left",
                   font=dict(size=12))

# Add slider for zooming
fig.update_layout(
    title=
    f"Actual vs Predicted Gold Price for the last {years_window} years with normalized factors",
    legend_title="Legend",
    xaxis=dict(
        rangeslider=dict(visible=True),
        title='Date',
        type="date",
        fixedrange=False  # allows zooming on x-axis
    ),
    yaxis=dict(
        title='Gold Price (USD)',
        fixedrange=True  # prevents zooming on y-axis
    ),
    dragmode='pan',
    template='plotly_dark')

# Save the plot to an HTML file if output is provided, otherwise show it in the browser
if args.output:
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)  # Ensure directory exists
    fig.write_html(args.output,
                   config={
                       'scrollZoom': True,
                       'modeBarButtonsToAdd': ['pan2d']
                   })
    print(f"Graph saved to {args.output}")
else:
    fig.show(config={'scrollZoom': True, 'modeBarButtonsToAdd': ['pan2d']})
