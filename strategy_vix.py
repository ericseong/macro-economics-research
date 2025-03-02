'''
This is a VIX-based contrarian trading strategy that:

Buys SPY when the market is fearful (high VIX).
Sells SPY when the market is calm (low VIX).
Tracks performance over time and visualizes the results.

Example usage:
  strategy_vix.py vix_low vix_high amount_trading
  strategy_vix.py 15 25 10 // we sell at vix_low and buy at vix_high. amount_trading is num. SPY stocks.
'''

import argparse
import yfinance as yf
#import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta


def simulate_trading(vix_low, vix_high, amount_trading):
    # Ensure amount_trading is float
    amount_trading = float(amount_trading)

    # Define simulation period (last 10 years)
    end_date = datetime.today()
    start_date = end_date - timedelta(days=5 * 365)

    # Fetch data
    #spy = yf.download("SPY", start=start_date, end=end_date)
    spy = yf.download("SPY", start=start_date, end=end_date, group_by="ticker")
    spy.columns = spy.columns.droplevel(0)  # Drop the top level (ticker)
    vix = yf.download("^VIX", start=start_date, end=end_date)["Close"]
    #print(spy.head())

    if spy.empty or vix.empty:
        print("Error: Unable to retrieve data.")
        return

    spy['VIX'] = vix
    spy = spy.dropna()  # Ensure we have only complete rows

    # Calculate moving averages
    spy['100_MA'] = spy['Close'].rolling(window=100).mean()
    spy['200_MA'] = spy['Close'].rolling(window=200).mean()

    # Initialize simulation variables
    cash = 300000  # Starting cash
    holdings = 0
    transactions = []
    fees = 0.002  # 0.2% trading fee

    # Monthly tracking variables
    monthly_summary = []

    # Simulation
    for i in range(len(spy)):
        date = spy.index[i]
        #vix_value = spy['VIX'][i]
        vix_value = spy['VIX'].iloc[i]
        #price = spy['Close'][i]
        #price = float(spy['Close'].iloc[i])
        price = spy['Close'].iloc[i].item()

        # Buy condition
        if vix_value > vix_high:
            buy_amount = amount_trading
            cost = buy_amount * price
            if cash >= cost:
                cash -= cost
                holdings += buy_amount
                transactions.append((date, "BUY", buy_amount, price))
                print(f"[{date}] BUY: Amount={buy_amount}, Price={price:.2f}")
            else:
                print(f"[{date}] BUY SKIPPED: Insufficient cash. Cash={cash:.2f}, Cost={cost:.2f}")

        # Sell condition
        elif vix_value < vix_low:
            sell_amount = min(holdings, amount_trading)
            if sell_amount > 0:
                revenue = sell_amount * price
                transaction_fee = revenue * fees
                cash += revenue - transaction_fee
                holdings -= sell_amount
                transactions.append((date, "SELL", sell_amount, price))
                print(f"[{date}] SELL: Amount={sell_amount}, Price={price:.2f}")
            else:
                print(f"[{date}] SELL SKIPPED: No holdings to sell.")

        # End-of-month snapshot
        if i == len(spy) - 1 or spy.index[i + 1].month != date.month:
            monthly_summary.append((date.to_period('M'), holdings, cash))

    # Print end-of-month stocks and cash
    print("\nEnd-of-Month Summary:")
    for month, stocks, cash in monthly_summary:
        print(f"{month}: Stocks Owned={stocks}, Cash Available=${cash:.2f}")

    # Print final results
    print("\nFinal Holdings:")
    print(f"Cash: ${cash:.2f}")
    print(f"Holdings: {holdings} shares of SPY")

    # Print transactions summary
    completed_transactions = sum(1 for t in transactions if t[1] == "SELL")
    print("\nTransaction Summary:")
    print(f"Total Transactions: {len(transactions)}")
    print(f"Completed SELL Transactions: {completed_transactions}")
    print(f"Remaining Holdings: {holdings}")

    # Visualization
    fig = go.Figure()

    # SPY price with red/green candlesticks
    fig.add_trace(go.Candlestick(
        x=spy.index,
        open=spy['Open'],
        high=spy['High'],
        low=spy['Low'],
        close=spy['Close'],
        name="SPY",
        increasing=dict(line=dict(color="red")),
        decreasing=dict(line=dict(color="green"))
    ))

    # 100-day moving average
    fig.add_trace(go.Scatter(
        x=spy.index,
        y=spy['100_MA'],
        mode='lines',
        name="100-Day MA",
        line=dict(color='blue', dash='solid')
    ))

    # 200-day moving average
    fig.add_trace(go.Scatter(
        x=spy.index,
        y=spy['200_MA'],
        mode='lines',
        name="200-Day MA",
        line=dict(color='orange', dash='solid')
    ))

    # VIX graph (right y-axis)
    fig.add_trace(go.Scatter(
        x=spy.index, y=spy['VIX'],
        mode='lines',
        name="VIX",
        line=dict(color='purple'),
        yaxis="y2"
    ))

    # Grouped Buy/Sell markers
    buy_dates = [date for date, action, amount, price in transactions if action == "BUY"]
    buy_prices = [price for date, action, amount, price in transactions if action == "BUY"]

    sell_dates = [date for date, action, amount, price in transactions if action == "SELL"]
    sell_prices = [price for date, action, amount, price in transactions if action == "SELL"]

    # Add a single trace for all Buy arrows
    fig.add_trace(go.Scatter(
        x=buy_dates, y=buy_prices,
        mode='markers',
        marker=dict(symbol='triangle-up', color='blue', size=12),
        name="Buy"  # Single legend entry for Buy
    ))

    # Add a single trace for all Sell arrows
    fig.add_trace(go.Scatter(
        x=sell_dates, y=sell_prices,
        mode='markers',
        marker=dict(symbol='triangle-down', color='red', size=12),
        name="Sell"  # Single legend entry for Sell
    ))

    # Update layout for dual axes
    fig.update_layout(
        title="Trading Simulation with SPY and VIX",
        xaxis_title="Date",
        yaxis=dict(
            title="SPY Price",
            side="left"
        ),
        yaxis2=dict(
            title="VIX",
            overlaying="y",
            side="right"
        ),
        legend=dict(x=0, y=1),
    )

    fig.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate trading based on VIX levels.")
    parser.add_argument("VIX_LOW", type=float, help="The lower threshold for VIX to sell SPY. ex. 15")
    parser.add_argument("VIX_HIGH", type=float, help="The upper threshold for VIX to buy SPY., ex. 25")
    parser.add_argument("AMOUNT_TRADING", type=float, help="The amount of SPY to trade each time., ex. 1")
    args = parser.parse_args()
    simulate_trading(args.VIX_LOW, args.VIX_HIGH, args.AMOUNT_TRADING)

