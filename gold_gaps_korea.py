'''
NAV gap analysis for the recent two years. We use GLD and ACE KRX 금현물 ETF for the analysis.
Usage: gold_gaps.py

'''

import os
import argparse
import yfinance as yf
from curl_cffi import requests
import pandas as pd
import plotly.graph_objects as go

# Argument parser for optional output filename
parser = argparse.ArgumentParser(
  description="Gold price gap analysis between Korea and US")
parser.add_argument("--years",
                    type=int,
                    default=2,
                    help="Number of years for data retrieval.")
parser.add_argument("--output",
                    type=str,
                    default=None,
                    help="Output file path for saving the HTML graph.")
args = parser.parse_args()

# Fetch 2 years of data
tickers = ["GLD", "411060.KS", "USDKRW=X"]
session = requests.Session(impersonate="chrome")
data = yf.download(tickers, period=f"{args.years}y", interval="1d", session=session)
print(data)

# Handle NaN for today's GLD price. It's to make up the missing today's price
# due to the time difference between Korea and USA.
if pd.isna(data.loc[data.index[-1], "Close"]["GLD"]):
    data.at[data.index[-1], ("Close", "GLD")] = data.at[data.index[-2], ("Close", "GLD")]
print(data)

# Extract OHLCV data
gld = data["Close"]["GLD"]
kr_gold = data["Close"]["411060.KS"]
usd_krw = data["Close"]["USDKRW=X"]
print(usd_krw)

# Drop rows with missing data
data_cleaned = pd.concat([gld, kr_gold, usd_krw], axis=1, join="inner").dropna()
data_cleaned.columns = ["GLD", "411060.KS", "USDKRW"]

# Convert 411060.KS price to USD
data_cleaned["411060_KS_USD"] = data_cleaned["411060.KS"] / data_cleaned["USDKRW"]

# Normalize values for visualization
gld_norm = (data_cleaned["GLD"] / data_cleaned["GLD"].iloc[0]) * 100
kr_gold_norm = (data_cleaned["411060_KS_USD"] / data_cleaned["411060_KS_USD"].iloc[0]) * 100

# Calculate ETF Premium/Discount as normalized difference
etf_premium_discount = 100 * (kr_gold_norm - gld_norm) / kr_gold_norm # in percentage

# Create figure
fig = go.Figure()

# Add line traces for normalized values
fig.add_trace(go.Scatter(
    x=data_cleaned.index, y=gld_norm, mode="lines",
      line=dict(color="gold", dash="solid"),
      name="GLD ETF Normalized (%)", yaxis="y1"
))
fig.add_trace(go.Scatter(
    x=data_cleaned.index, y=kr_gold_norm, mode="lines",
      line=dict(color="red", dash="solid"),
      name="411060.KS (ACE KRX 금현물) Normalized (%)", yaxis="y1"
))

# Add bar chart for ETF Premium/Discount
fig.add_trace(go.Bar(
    x=data_cleaned.index, y=etf_premium_discount, name="Korea market premium Normalized (%)", yaxis="y2", marker=dict(color="gray")
))

# Layout settings
fig.update_layout(
    title="Korea gold price premium analysis",
    xaxis=dict(
      rangeslider=dict(visible=True),
      type="date",
      title="Date",
      fixedrange=False # allows zooming on x-axis
    ),
    yaxis=dict(
      title="Normalized price",
      side="left",
      fixedrange=True # prevent zooming on y-axis
    ),
    yaxis2=dict(
      title="Premium",
      overlaying="y",
      side="right",
      showgrid=False,
      fixedrange=True # prevents zooming on secondary y-axis
    ),
    legend=dict(x=0, y=1),
    dragmode='pan',
    template="plotly_dark"
)

# Save the plot to an HTML file if output is provided, otherwise show it in the browser
if args.output:
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)  # Ensure directory exists
    fig.write_html(args.output, config={'scrollZoom': True, 'modeBarButtonsToAdd': ['pan2d']})
    print(f"Graph saved to {args.output}")
else:
    fig.show(config={'scrollZoom': True, 'modeBarButtonsToAdd': ['pan2d']})

# eof

