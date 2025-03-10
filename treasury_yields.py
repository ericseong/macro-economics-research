'''
treasury_yields.py

Known issue:
  - Data for the most recent date is missing for 10-year US treasury data
  fetched by fredapi.
'''
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from fredapi import Fred
from datetime import datetime, timedelta
import warnings
import yfinance as yf

# Print warning message with clickable URL
warnings.warn(
    "Without downloading Japan 10-year treasury yield data manually, "
    "the chart is to be drawn with outdated data! You can download the data from "
    "https://www.wsj.com/market-data/quotes/bond/BX/TMBMKJP-10Y/historical-prices"
    "https://www.wsj.com/market-data/quotes/bond/BX/TMUBMUSD10Y/historical-prices"
    "Make sure the name of the data is to be JP.csv and US.csv",
    category=UserWarning
)

# Prompt user for confirmation
response = input("The Japan 10-year treasury yield data might be outdated. Continue anyway? (yes/no): ").strip().lower()
if response not in ['yes', 'y']:
    print("Execution stopped. Please update the Japan 10-year treasury yield data and try again.")
    exit()

'''
# Read FRED API key
with open('credentials/credential_fred_api.txt', 'r') as file:
    fred_api_key = file.read().strip()

# Initialize FRED
fred = Fred(api_key=fred_api_key)

# Get US 10-year Treasury yield data (series ID: DGS10)
us_treasury = fred.get_series('DGS10', start_date, end_date)
print('us_treasury')
print(us_treasury)
us_treasury = us_treasury.dropna()

# Create OHLC data for US
us_treasury_df = pd.DataFrame({
    'Date': us_treasury.index,
    'Close': us_treasury.values
})
us_treasury_df['Open'] = us_treasury_df['Close'].shift(1, fill_value=us_treasury_df['Close'].iloc[0])
us_treasury_df['High'] = us_treasury_df['Close'] * 1.005
us_treasury_df['Low'] = us_treasury_df['Close'] * 0.995
'''

end_date = datetime.now().date()
start_date = end_date - timedelta(days=365*10)  # 10 years of data

# Read Japan 10-year Treasury yield from CSV
japan_treasury = pd.read_csv('JP.csv')
print('japan_treasury')
print(japan_treasury)
japan_treasury.columns = japan_treasury.columns.str.strip()
japan_treasury['Date'] = pd.to_datetime(japan_treasury['Date'], format='%m/%d/%y')
japan_treasury = japan_treasury.sort_values('Date')

# Read Japan 10-year Treasury yield from CSV
us_treasury = pd.read_csv('US.csv')
print('us_treasury')
print(us_treasury)
us_treasury.columns = us_treasury.columns.str.strip()
us_treasury['Date'] = pd.to_datetime(us_treasury['Date'], format='%m/%d/%y')
us_treasury = us_treasury.sort_values('Date')

# Get NASDAQ data using yfinance
nasdaq = yf.download('^IXIC', start=start_date, end=end_date, auto_adjust=False)
# Reset index and rename columns explicitly
nasdaq = nasdaq.reset_index()
nasdaq = nasdaq[['Date', 'Open', 'High', 'Low', 'Close']]
nasdaq.columns = ['Date', 'Open_NQ', 'High_NQ', 'Low_NQ', 'Close_NQ']  # Rename to avoid conflicts

# Merge the dataframes
merged_df = pd.merge_asof(
    us_treasury,
    japan_treasury[['Date', 'Open', 'High', 'Low', 'Close']],
    on='Date',
    direction='nearest',
    suffixes=('_US', '_JP')
)

merged_df = pd.merge_asof(
    merged_df,
    nasdaq[['Date', 'Open_NQ', 'High_NQ', 'Low_NQ', 'Close_NQ']],
    on='Date',
    direction='nearest'
)

# Calculate daily gap using closing prices
merged_df['Yield_Gap'] = merged_df['Close_US'] - merged_df['Close_JP']

# Create color condition for bars
colors = ['yellow' if x < 3 else 'lightgray' for x in merged_df['Yield_Gap']]

# Create the plot with subplots
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Add US Treasury yield candlestick
fig.add_trace(
    go.Candlestick(
        x=merged_df['Date'],
        open=merged_df['Open_US'],
        high=merged_df['High_US'],
        low=merged_df['Low_US'],
        close=merged_df['Close_US'],
        name='US 10Y Yield',
        increasing_line_color='red',
        decreasing_line_color='green'
    ),
    secondary_y=False,
)

# Add Japan Treasury yield candlestick
fig.add_trace(
    go.Candlestick(
        x=merged_df['Date'],
        open=merged_df['Open_JP'],
        high=merged_df['High_JP'],
        low=merged_df['Low_JP'],
        close=merged_df['Close_JP'],
        name='Japan 10Y Yield',
        increasing_line_color='red',
        decreasing_line_color='green'
    ),
    secondary_y=False,
)

# Add yield gap bar
fig.add_trace(
    go.Bar(
        x=merged_df['Date'],
        y=merged_df['Yield_Gap'],
        name='Yield Gap (US-JP)',
        opacity=0.5,
        marker_color=colors
    ),
    secondary_y=False,
)

# Add NASDAQ candlestick
fig.add_trace(
    go.Candlestick(
        x=merged_df['Date'],
        open=merged_df['Open_NQ'],
        high=merged_df['High_NQ'],
        low=merged_df['Low_NQ'],
        close=merged_df['Close_NQ'],
        name='NASDAQ',
        increasing_line_color='orange',
        decreasing_line_color='blue'
    ),
    secondary_y=True,
)

# Update layout
fig.update_layout(
    title='US vs Japan 10-Year Treasury Yields, Yield Gap, and NASDAQ',
    #xaxis_title='Date',
    hovermode='x unified',
    xaxis=dict(
      rangeslider=dict(visible=True),
      title='Date',
      type='date',
      fixedrange=False # allows zooming on x-axis
    ),
    yaxis=dict(
      fixedrange=True # prevents zooming on y-axis
    ),
    showlegend=True,
    dragmode='pan',
    template='plotly_dark'
)

# Update y-axes
fig.update_yaxes(
    title_text='Yield (%) / Yield Gap (%)',
    secondary_y=False
)
fig.update_yaxes(
    title_text='NASDAQ Index',
    secondary_y=True
)

# Show the plot
fig.show(config={'scrollZoom': True, 'modeBarButtonsToAdd': ['pan2d']})

