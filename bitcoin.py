import ccxt
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from plotly.subplots import make_subplots
import numpy as np
from sklearn.linear_model import LinearRegression

# Function to fetch historical data using CCXT from Kraken exchange
def fetch_historical_data(exchange, symbol, timeframe, since):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since)
    data = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    data.set_index('timestamp', inplace=True)
    return data

# Initialize the exchange (Kraken provides historical data back to 2011 for BTC/USD)
exchange = ccxt.kraken()
symbol = "BTC/USD"
timeframe = '1w'  # Weekly data
since = exchange.parse8601('2014-01-01T00:00:00Z')  # Fetch data from 2014 onwards

# Fetch data
try:
    data = fetch_historical_data(exchange, symbol, timeframe, since)
except Exception as e:
    print(f"Error fetching data: {e}")
    exit()

# Define Bitcoin halving events
halving_dates = [
    datetime(2012, 11, 28),
    datetime(2016, 7, 9),
    datetime(2020, 5, 11),
    datetime(2024, 4, 23),  # Expected date
    datetime(2028, 5, 15)   # Expected future halving
]

# Prepare data for linear regression on 'close' prices
data['ordinal_date'] = data.index.map(datetime.toordinal)  # Convert dates to ordinal numbers
x = data['ordinal_date'].values.reshape(-1, 1)
y = np.log(data['close'].values.reshape(-1, 1))  # Use log-transformed prices for better regression
model = LinearRegression()
model.fit(x, y)

# Extend regression line to the end of the X-axis range
x_future = pd.date_range(start=data.index[0], end=datetime(2034, 12, 31), freq='W').map(datetime.toordinal).values.reshape(-1, 1)
y_future_pred = model.predict(x_future)

# Create subplots for OHLC and Volume
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                    row_heights=[0.7, 0.3],
                    subplot_titles=('BTC/USD Weekly OHLC (Log Scale)', 'Volume'))

# Add OHLC data
fig.add_trace(go.Candlestick(
    x=data.index,
    open=data['open'],
    high=data['high'],
    low=data['low'],
    close=data['close'],
    name='OHLC'
), row=1, col=1)

# Add regression line (exponentiated to revert log transformation)
fig.add_trace(go.Scatter(
    x=pd.to_datetime([datetime.fromordinal(int(date)) for date in x_future.flatten()]),
    y=np.exp(y_future_pred.flatten()),
    mode='lines',
    name='Linear Regression',
    line=dict(color='yellow', dash='dash')
), row=1, col=1)

# Add volume data
fig.add_trace(go.Bar(
    x=data.index,
    y=data['volume'],
    name='Volume',
    marker_color='blue'
), row=2, col=1)

# Add vertical lines for halving events
for date in halving_dates:
    if datetime(2014, 1, 1) <= date <= datetime(2034, 12, 31):
        fig.add_vline(x=date, line_width=1, line_dash="dot", line_color="red")
        fig.add_annotation(
            x=date,
            y=1,
            xref="x",
            yref="paper",
            showarrow=False,
            text="Halving",
            font=dict(color="red")
        )

# Update layout
fig.update_layout(
    title='BTC/USD Weekly OHLC and Volume with Halving Events (Log Scale, Kraken)',
    xaxis_title='Date',
    yaxis_title='Price (USD, Log Scale)',
    yaxis_type='log',  # Set the Y-axis to logarithmic scale
    yaxis_range=[2, 7],  # Log scale range: 10^2 (100) to 10^7 (10,000,000)
    xaxis_range=[datetime(2014, 1, 1), datetime(2034, 12, 31)],  # Set the X-axis range to 2014-2034
    xaxis_showgrid=True,  # Enable horizontal grid lines
    yaxis_showgrid=True,  # Enable vertical grid lines for OHLC chart
    template='plotly_dark',  # Restore previous dark background template
    showlegend=True
)

# Save the chart as an HTML file
fig.write_html("btc_kraken_weekly_chart_with_log_scale_and_extended_y.html")

print("Chart saved as btc_kraken_weekly_chart_with_log_scale_and_extended_y.html")

