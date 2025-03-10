import argparse
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pandas as pd

def parse_arguments():
    parser = argparse.ArgumentParser(description='Currency comparison chart generator')
    parser.add_argument('currencies', nargs='+', help='Currency codes (e.g., EUR JPY)')
    parser.add_argument('years', type=int, help='Number of years of data to fetch')
    parser.add_argument('--output', default=None,
                       help='Output file path (if not provided, shows in browser)')
    return parser.parse_args()

def fetch_currency_data(currencies, years):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)

    currencies = [curr.upper() for curr in currencies]
    currencies = ['JPY' if curr == 'JPN' else curr for curr in currencies]

    symbols = [f"{curr}=X" for curr in currencies] + ["DX-Y.NYB"]
    data = {}

    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date)
            if not df.empty and 'Close' in df.columns:
                # For currency pairs, invert the values to show DXY/currency
                if symbol != "DX-Y.NYB":
                    data[symbol] = 1 / df['Close'].dropna()  # Invert the currency pair
                else:
                    data[symbol] = df['Close'].dropna()
            else:
                print(f"No valid data for {symbol}")
        except Exception as e:
            print(f"Error fetching data for {symbol}: {str(e)}")

    return data

def normalize_data(data):
    normalized = {}
    if "DX-Y.NYB" not in data or data["DX-Y.NYB"].empty:
        return normalized

    # Get the starting value of Dollar Index
    dxy_start = data["DX-Y.NYB"].iloc[0]

    for symbol, series in data.items():
        if not series.empty and len(series) > 0:
            # Normalize all series to DXY's starting value
            normalized[symbol] = (series / series.iloc[0]) * dxy_start
    return normalized

def create_chart(normalized_data, output_file=None):
    fig = go.Figure()

    for symbol, series in normalized_data.items():
        # Change label format from currency/USD to DXY/currency
        base_label = f"DXY/{symbol.replace('=X', '')}" if '=X' in symbol else 'DXY'
        label = f"{base_label}"
        if symbol == "DX-Y.NYB":
            fig.add_trace(
                go.Scatter(
                    x=series.index,
                    y=series.values,
                    name=label,
                    hovertemplate='%{y:.2f}<br>Date: %{x|%Y-%m-%d}',
                    line=dict(color='white', dash='dot', width=1)
                )
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=series.index,
                    y=series.values,
                    name=label,
                    hovertemplate='%{y:.2f}<br>Date: %{x|%Y-%m-%d}'
                )
            )

    fig.update_layout(
        title='Currency Comparison (Normalized to DXY Start Value)',
        legend_title='Currencies',
        hovermode='x unified',
        xaxis=dict(
            rangeslider=dict(visible=True),
            title='Date',
            type='date',
            fixedrange=False # allows zooming on x-axis
        ),
        yaxis=dict(
          title='Value (Normalized to DXY Starting Point)',
          fixedrange=True # prevents zooming on y-axis
        ),
        dragmode='pan',
        template='plotly_dark'
    )

    if output_file:
        fig.write_html(output_file, config={'scrollZoom': True, 'modeBarButtonsToAdd': ['pan2d']})
        print(f"Chart saved as {output_file}")
    else:
        fig.show(config={'scrollZoom': True, 'modeBarButtonsToAdd': ['pan2d']})

def main():
    args = parse_arguments()

    print(f"Fetching {args.years} years of data for {args.currencies}")
    raw_data = fetch_currency_data(args.currencies, args.years)
    print(raw_data)

    if not raw_data:
        print("No data fetched. Exiting.")
        return

    normalized_data = normalize_data(raw_data)

    if not normalized_data:
        print("No data to plot after normalization. Exiting.")
        return

    create_chart(normalized_data, args.output)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {str(e)}")
