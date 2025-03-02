'''
NAV gap analysis for the recent two years. We use GLD and ACE KRX 금현물 ETF for the analysis.
Usage: gold_gaps.py

'''

import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

# Fetch 2 years of data
tickers = ["GLD", "411060.KS", "USDKRW=X"]
data = yf.download(tickers, period="2y", interval="1d")
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
etf_premium_discount = kr_gold_norm - gld_norm

# Create figure
fig = go.Figure()

# Add line traces for normalized values
fig.add_trace(go.Scatter(
    x=data_cleaned.index, y=gld_norm, mode="lines", name="GLD Normalized (%)", yaxis="y1"
))
fig.add_trace(go.Scatter(
    x=data_cleaned.index, y=kr_gold_norm, mode="lines", name="411060.KS Normalized (%)", yaxis="y1"
))

# Add bar chart for ETF Premium/Discount
fig.add_trace(go.Bar(
    x=data_cleaned.index, y=etf_premium_discount, name="ETF Premium/Discount (Normalized)", yaxis="y2", marker=dict(color="gray")
))

# Layout settings
fig.update_layout(
    title="ETF Premium/Discount vs NAV (GLD) - 2 Years",
    xaxis=dict(
        rangeslider=dict(visible=True), type="date", title="Date"
    ),
    yaxis=dict(title="Normalized Value (%)", side="left"),
    yaxis2=dict(
        title="ETF Premium/Discount (Normalized)", overlaying="y", side="right", showgrid=False
    ),
    legend=dict(x=0, y=1),
    template="plotly_dark"
)

# Show figure
fig.show()

