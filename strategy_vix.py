import argparse
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pandas as pd

def download_data(ticker, start_date, end_date, retry_count=5, column=None):
    if isinstance(start_date, str):
        _start_date = datetime.strptime(start_date, "%Y-%m-%d")
    else:
        _start_date = start_date
    end_date = datetime.strptime(end_date, "%Y-%m-%d") if isinstance(end_date, str) else end_date

    attempt = 0

    while attempt <= retry_count:
        try:
            print(f"Attempt {attempt + 1}: Downloading {ticker} from {_start_date} to {end_date}")
            data = yf.download(ticker, start=_start_date, end=end_date)

            if data.empty:
                raise ValueError(f"No data returned for {ticker}, possibly no trading data for this date.")

            print(f"Columns for {ticker}: {data.columns.tolist()}")

            if column:
                if isinstance(data.columns, pd.MultiIndex):
                    level_0_names = data.columns.get_level_values(0).tolist()
                    level_1_names = data.columns.get_level_values(1).tolist()
                    if ticker in level_1_names:
                        if (column, ticker) in data.columns:
                            return data[(column, ticker)]
                        else:
                            raise KeyError(f"Column {column} not found in (Price, Ticker) multi-level index for {ticker}")
                    elif ticker in level_0_names:
                        if (ticker, column) in data.columns:
                            return data[(ticker, column)]
                        else:
                            raise KeyError(f"Column {column} not found in (Ticker, Price) multi-level index for {ticker}")
                    else:
                        raise KeyError(f"Ticker {ticker} not found in multi-level index")
                else:
                    if column in data.columns:
                        return data[column]
                    else:
                        raise KeyError(f"Column {column} not found in single-level index for {ticker}")
            return data

        except Exception as e:
            print(f"Failed to download {ticker}: {e}")
            if attempt == retry_count:
                print(f"Max retries ({retry_count}) reached for {ticker}. Returning empty data.")
                return None
            _start_date -= timedelta(days=1)
            attempt += 1

def simulate_trading(vix_low, vix_high, amount_trading, output_file=None):
    amount_trading = float(amount_trading)
    end_date = datetime.today()
    start_date = end_date - timedelta(days=4 * 365)

    # Download SPY data
    spy = download_data("SPY", start_date, end_date)
    if spy is not None:
      if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.droplevel(1)
      spy['100_MA'] = spy['Close'].rolling(window=100, min_periods=100).mean()
      spy['200_MA'] = spy['Close'].rolling(window=200, min_periods=200).mean()
      print('spy (before merging VIX):')
      print(spy.tail())
      print(f"SPY columns after processing: {spy.columns.tolist()}")
    else:
        spy = None  # Ensure spy is None if download fails

    # Download VIX data (only "Close" column)
    vix = download_data("^VIX", start_date, end_date, column="Close")
    #vix = download_data("^VIX", start_date, end_date)
    if vix is None:
      exit(1)
    print('vix:')
    print(vix)

    # Merge VIX into SPY DataFrame
    if spy is not None and vix is not None:
        spy['VIX'] = vix  # Add VIX Close prices as a new column in spy
        print('spy (after merging VIX):')
        print(spy.tail())
        print(f"Final SPY columns: {spy.columns.tolist()}")

    # Check if either dataset is empty or None
    if spy is None or vix is None or spy.empty or vix.empty:
        print("Error: Unable to retrieve data.")
        # return  # Uncomment this if this code is inside a function
    else:
        print("Data retrieval successful!")

    cash = 300000
    holdings = 0
    transactions = []
    fees = 0.002
    monthly_summary = []

    for i in range(len(spy)):
        date = spy.index[i]
        vix_value = spy['VIX'].iloc[i]
        price = spy['Close'].iloc[i].item()

        if vix_value > vix_high:
            buy_amount = amount_trading
            cost = buy_amount * price
            if cash >= cost:
                cash -= cost
                holdings += buy_amount
                transactions.append((date, "BUY", buy_amount, price))
                print(f"[{date}] BUY: Amount={buy_amount}, Price={price:.2f}")
            else:
                print(
                    f"[{date}] BUY SKIPPED: Insufficient cash. Cash={cash:.2f}, Cost={cost:.2f}"
                )

        elif vix_value < vix_low:
            sell_amount = min(holdings, amount_trading)
            if sell_amount > 0:
                revenue = sell_amount * price
                transaction_fee = revenue * fees
                cash += revenue - transaction_fee
                holdings -= sell_amount
                transactions.append((date, "SELL", sell_amount, price))
                print(
                    f"[{date}] SELL: Amount={sell_amount}, Price={price:.2f}")
            else:
                print(f"[{date}] SELL SKIPPED: No holdings to sell.")

        if i == len(spy) - 1 or spy.index[i + 1].month != date.month:
            monthly_summary.append((date.to_period('M'), holdings, cash))

    print("\nEnd-of-Month Summary:")
    for month, stocks, cash in monthly_summary:
        print(f"{month}: Stocks Owned={stocks}, Cash Available=${cash:.2f}")

    print("\nFinal Holdings:")
    print(f"Cash: ${cash:.2f}")
    print(f"Holdings: {holdings} shares of SPY")

    completed_transactions = sum(1 for t in transactions if t[1] == "SELL")
    print("\nTransaction Summary:")
    print(f"Total Transactions: {len(transactions)}")
    print(f"Completed SELL Transactions: {completed_transactions}")
    print(f"Remaining Holdings: {holdings}")

    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(x=spy.index,
                       open=spy['Open'],
                       high=spy['High'],
                       low=spy['Low'],
                       close=spy['Close'],
                       name="SPY",
                       increasing=dict(line=dict(color="red"),
                                       fillcolor="red"),
                       decreasing=dict(line=dict(color="green"),
                                       fillcolor="green")))

    fig.add_trace(
        go.Scatter(x=spy.index,
                   y=spy['100_MA'],
                   mode='lines',
                   name="100-Day MA",
                   line=dict(color='lightgray', dash='dot', width=2)))
    fig.add_trace(
        go.Scatter(x=spy.index,
                   y=spy['200_MA'],
                   mode='lines',
                   name="200-Day MA",
                   line=dict(color='lightgray', dash='dot', width=4)))
    fig.add_trace(
        go.Scatter(x=spy.index,
                   y=spy['VIX'],
                   mode='lines',
                   name="VIX",
                   line=dict(color='orchid'),
                   yaxis="y2"))

    buy_dates = [
        date for date, action, amount, price in transactions if action == "BUY"
    ]
    buy_prices = [
        price for date, action, amount, price in transactions
        if action == "BUY"
    ]
    sell_dates = [
        date for date, action, amount, price in transactions
        if action == "SELL"
    ]
    sell_prices = [
        price for date, action, amount, price in transactions
        if action == "SELL"
    ]

    fig.add_trace(
        go.Scatter(x=buy_dates,
                   y=buy_prices,
                   mode='markers',
                   marker=dict(symbol='triangle-up', color='white', size=8),
                   name="Buy"))
    fig.add_trace(
        go.Scatter(
            x=sell_dates,
            y=sell_prices,
            mode='markers',
            marker=dict(
                symbol='triangle-down',
                color='rgba(0,0,0,0)',  # transparent fill
                size=8,
                line=dict(color='yellow', width=2)),
            name="Sell"))
    fig.update_layout(
        title="Trading Simulation with SPY and VIX",
        xaxis=dict(
            rangeslider=dict(visible=True),
            type='date',
            fixedrange=False  # allow zooming on x-axis
        ),
        #xaxis_title="Date",
        yaxis=dict(
            title="SPY Price",
            side="left",
            fixedrange=True  # prevents zooming on y-axis
        ),
        yaxis2=dict(
            title="VIX",
            overlaying="y",
            side="right",
            fixedrange=True  # prevents zooming on secondary y-axis
        ),
        legend=dict(x=0, y=1),
        dragmode='pan',
        template='plotly_dark')

    if output_file:
        fig.write_html(output_file,
                       config={
                           'scrollZoom': True,
                           'modeBarButtonsToAdd': ['pan2d'],
                           'modeBarButtonsToRemove': ['zoom2d']
                       })
        print(f"Graph saved to {output_file}")
    else:
        fig.show(
            config={
                'scrollZoom': True,
                'modeBarButtonsToAdd': ['pan2d'],
                'modeBarButtonsToRemove': ['zoom2d']
            })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simulate trading based on VIX levels.")
    parser.add_argument("VIX_LOW",
                        type=float,
                        help="The lower threshold for VIX to sell SPY. ex. 15")
    parser.add_argument("VIX_HIGH",
                        type=float,
                        help="The upper threshold for VIX to buy SPY., ex. 25")
    parser.add_argument("AMOUNT_TRADING",
                        type=float,
                        help="The amount of SPY to trade each time., ex. 1")
    parser.add_argument("--output",
                        type=str,
                        default=None,
                        help="Output HTML file for the graph.")
    args = parser.parse_args()
    simulate_trading(args.VIX_LOW, args.VIX_HIGH, args.AMOUNT_TRADING,
                     args.output)
