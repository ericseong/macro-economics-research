# WIP!!!!!!

import os
import argparse
import pandas as pd
#import requests
import plotly.graph_objects as go
import yfinance as yf
import pytz
from datetime import datetime, timedelta
from curl_cffi import requests

# Define Korean Standard Time (KST)
KST = pytz.timezone("Asia/Seoul")

# Argument parser
parser = argparse.ArgumentParser(description="Bitcoin price gap analysis between Korea and US")
parser.add_argument("--output", type=str, default=None, help="Output file path for saving the HTML graph.")
parser.add_argument("--years", type=int, default=2, help="Number of years of historical data to fetch.")
parser.add_argument("--dataprovider", type=str, default='binance', help="Where to fetch US bitcoin price.")
args = parser.parse_args()

# Compute the total days based on the provided year span
days_to_fetch = args.years * 365


def get_yfinance_btc_price(days_to_fetch=730):  # Supports up to several years
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days_to_fetch)).strftime('%Y-%m-%d')

    ticker = "BTC-USD"
    session = requests.Session(impersonate="chrome")
    df = yf.download(ticker, start=start_date, end=end_date, interval="1d", session=session)

    if df.empty:
        raise ValueError("No data received from Yahoo Finance")

    # Convert index to KST
    df.index = df.index.tz_localize("UTC").tz_convert(KST)

    return df[["Close"]].rename(columns={"Close": "close"}).sort_index()


def get_coingecko_btc_price(days_to_fetch=1000):
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params = {"vs_currency": "usd", "days": days_to_fetch, "interval": "daily"}
    response = requests.get(url, params=params)

    if response.status_code != 200:
        raise Exception(f"Error fetching CoinGecko data: {response.status_code}, {response.text}")

    data = response.json()

    if "prices" not in data:
        raise ValueError(f"Unexpected CoinGecko response format: {data}")

    print("Data received from CoinGecko:", data["prices"][:2])  # Print first 2 rows

    df = pd.DataFrame(data["prices"], columns=["timestamp", "close"])

    # Convert timestamp from UTC to KST
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms").dt.tz_localize("UTC").dt.tz_convert(KST)
    df.set_index("timestamp", inplace=True)
    df["close"] = df["close"].astype(float)

    return df.sort_index()


def get_binance_btc_price(days_to_fetch=1000):
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": "BTCUSDT", "interval": "1d", "limit": min(days_to_fetch, 1000)}
    response = requests.get(url, params=params)
    print(f"Response from binance - Status Code: {response.status_code}, Response: {response.text}")

    if response.status_code != 200:
        raise Exception(f"Error fetching Binance data: {response.status_code}, {response.text}")

    data = response.json()

    # Validate the data structure before proceeding
    if not isinstance(data, list) or len(data) == 0 or not isinstance(data[0], list):
        raise ValueError(f"Unexpected Binance response format: {data}")

    print("data received from binance:", data[:2])  # Print only the first 2 rows to check structure

    # Define Binance Kline column names explicitly
    expected_columns = [
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "num_trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ]

    # Create DataFrame with proper column names
    df = pd.DataFrame(data, columns=expected_columns)

    # Convert timestamp from UTC to KST
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms").dt.tz_localize("UTC").dt.tz_convert(KST)
    df.set_index("timestamp", inplace=True)

    # Convert relevant columns to float
    numeric_cols = ["open", "high", "low", "close", "volume", "quote_asset_volume", "taker_buy_base", "taker_buy_quote"]
    df[numeric_cols] = df[numeric_cols].astype(float)

    # Ensure unique timestamps
    df = df[~df.index.duplicated(keep="first")]

    return df[["close"]].sort_index()

# Function to fetch BTC/KRW price from Upbit and keep timestamps in KST
def get_upbit_btc_price(days_to_fetch=1000):
    url = "https://api.upbit.com/v1/candles/days"
    headers = {"Accept": "application/json"}

    all_data = []
    count = 200
    to_date = datetime.now(KST)  # Start from today in KST

    while len(all_data) < days_to_fetch:
        params = {
            "market": "KRW-BTC",
            "count": min(count, days_to_fetch - len(all_data)),
            "to": to_date.strftime("%Y-%m-%dT%H:%M:%S")
        }
        response = requests.get(url, params=params, headers=headers)
        data = response.json()

        if not isinstance(data, list) or len(data) == 0:
            break

        all_data.extend(data)
        to_date = datetime.strptime(data[-1]["candle_date_time_utc"], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=pytz.utc).astimezone(KST)

    # Convert to DataFrame
    df = pd.DataFrame(all_data)

    # Keep timestamps in KST
    df["timestamp"] = pd.to_datetime(df["candle_date_time_kst"]).dt.tz_localize(KST)

    df.set_index("timestamp", inplace=True)
    df["close"] = df["trade_price"].astype(float)

    # Ensure unique timestamps
    df = df[~df.index.duplicated(keep="first")]

    return df[["close"]].sort_index()

# Function to fetch USD/KRW exchange rate from Yahoo Finance and convert to KST
def get_usd_krw(days_to_fetch=1000):
    ticker = "USDKRW=X"
    session = requests.Session(impersonate="chrome")
    start_date = (datetime.now() - timedelta(days=days_to_fetch)).strftime('%Y-%m-%d')
    data = yf.download(ticker, start=start_date, interval="1d", session=session)

    # Convert timestamps to KST
    data.index = data.index.tz_localize("UTC").tz_convert(KST)

    # Ensure unique timestamps
    data = data[~data.index.duplicated(keep="first")]

    return data["Close"]

'''
# Replace Binance function with CoinGecko
btc_usd = get_coingecko_btc_price()
'''

# Fetch Data
if args.dataprovider == 'coingecko':
  btc_usd = get_coingecko_btc_price(days_to_fetch)
elif args.dataprovider == 'yfinance':
  btc_usd = get_yfinance_btc_price(days_to_fetch)
else:
  btc_usd = get_binance_btc_price(days_to_fetch)

btc_krw = get_upbit_btc_price(days_to_fetch)
usd_krw = get_usd_krw(days_to_fetch)

# Merge datasets using KST timestamps
data_cleaned = pd.concat([btc_usd, btc_krw, usd_krw], axis=1, join="inner").dropna()
data_cleaned.columns = ["BTC_USD", "BTC_KRW", "USDKRW"]

# Forward-fill missing exchange rates for consistency
data_cleaned["USDKRW"] = data_cleaned["USDKRW"].ffill()

# Convert BTC/KRW to USD using the same day's exchange rate
data_cleaned["BTC_KRW_USD"] = data_cleaned["BTC_KRW"] / data_cleaned["USDKRW"]

# Normalize for visualization
btc_usd_norm = (data_cleaned["BTC_USD"] / data_cleaned["BTC_USD"].iloc[0]) * 100
btc_krw_norm = (data_cleaned["BTC_KRW_USD"] / data_cleaned["BTC_KRW_USD"].iloc[0]) * 100

# Calculate Kimchi Premium
btc_premium_discount = 100 * (btc_krw_norm - btc_usd_norm) / btc_krw_norm

# Create figure
fig = go.Figure()

# Add line traces for normalized values
fig.add_trace(go.Scatter(
    x=data_cleaned.index, y=btc_usd_norm, mode="lines",
    line=dict(color="yellow", dash="solid"),
    name="BTC/USD (Binance) Normalized (%)", yaxis="y1"
))
fig.add_trace(go.Scatter(
    x=data_cleaned.index, y=btc_krw_norm, mode="lines",
    line=dict(color="red", dash="solid"),
    name="BTC/KRW (Upbit) Normalized (%)", yaxis="y1"
))

# Add bar chart for Premium/Discount
fig.add_trace(go.Bar(
    x=data_cleaned.index, y=btc_premium_discount,
    name="Korea BTC Premium (%)", yaxis="y2",
    marker=dict(color="gray")
))

# Layout settings
fig.update_layout(
    title=f"Korea Bitcoin Premium Analysis (Upbit vs {args.dataprovider}) - Last {args.years} Years",
    xaxis=dict(
        rangeslider=dict(visible=True),
        type="date",
        title="Date",
        fixedrange=False
    ),
    yaxis=dict(
        title="Normalized Price",
        side="left",
        fixedrange=True
    ),
    yaxis2=dict(
        title="Premium (%)",
        overlaying="y",
        side="right",
        showgrid=False,
        fixedrange=True
    ),
    legend=dict(x=0, y=1),
    dragmode='pan',
    template="plotly_dark"
)

# Save or show the plot
if args.output:
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)  # Ensure directory exists
    fig.write_html(args.output, config={'scrollZoom': True, 'modeBarButtonsToAdd': ['pan2d']})
    print(f"Graph saved to {args.output}")
else:
    fig.show(config={'scrollZoom': True, 'modeBarButtonsToAdd': ['pan2d']})

# eof

