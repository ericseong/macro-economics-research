"""
This script retrieves performance data using Yahoo Finance (yfinance).
Performance data includes:
- Quarterly revenue, operating margin, and net margin.
- Trailing P/E and Forward P/E per quarter.
- Institutional holders' details.

**Usage:**
    python perf.py <symbol>

Example:
    python perf.py AAPL
"""

import yfinance as yf
from curl_cffi import requests
import sys
import pandas as pd

def format_number(value):
    """Format large numbers with suffixes like B (billion), M (million), and K (thousand)."""
    try:
        value = float(value)
        if abs(value) >= 1e9:
            return f"{value / 1e9:.2f}B$"
        elif abs(value) >= 1e6:
            return f"{value / 1e6:.2f}M$"
        elif abs(value) >= 1e3:
            return f"{value / 1e3:.2f}K$"
        else:
            return f"{value:.2f}$"
    except (ValueError, TypeError):
        return "N/A"

def get_financial_data(stock_symbol):
    """Retrieve and display financial data for the given stock symbol."""
    try:
        session = requests.Session(impersonate="chrome")
        stock = yf.Ticker(stock_symbol, session=session)

        # Fetch financials
        income_stmt = stock.quarterly_financials
        if income_stmt.empty:
            print(f"Error: No financial data available for {stock_symbol}.")
            return

        # Dynamically adjust to available quarters
        available_columns = income_stmt.columns[:min(len(income_stmt.columns), 5)]  # Use up to 5 most recent quarters

        revenue = income_stmt.loc['Total Revenue', available_columns] if 'Total Revenue' in income_stmt.index else pd.Series([0] * len(available_columns), index=available_columns)
        op_profit = income_stmt.loc['Operating Income', available_columns] if 'Operating Income' in income_stmt.index else revenue * 0
        net_profit = income_stmt.loc['Net Income', available_columns] if 'Net Income' in income_stmt.index else revenue * 0

        # Fetch stock info
        info = stock.info
        trailing_pe = info.get('trailingPE', 'N/A')
        forward_pe = info.get('forwardPE', 'N/A')
        trailing_eps = info.get('trailingEps', 'N/A')
        forward_eps = info.get('forwardEps', 'N/A')
        current_price = info.get('currentPrice', None)  # Stock price needed for trailing P/E
        total_shares = info.get('sharesOutstanding', None)  # Needed for per-share calculation

        # Calculate profit margins
        op_profit_ratio = [(op / rev) * 100 if rev else 'N/A' for op, rev in zip(op_profit, revenue)]
        net_profit_ratio = [(net / rev) * 100 if rev else 'N/A' for net, rev in zip(net_profit, revenue)]

        # Estimate quarterly EPS (Earnings Per Share)
        if total_shares and total_shares > 0:
            quarterly_eps = net_profit / total_shares
        else:
            quarterly_eps = pd.Series(["N/A"] * len(available_columns), index=available_columns)

        # Calculate Trailing P/E per quarter
        if current_price:
            quarterly_trailing_pe = [round(current_price / eps, 2) if eps and eps > 0 else 'N/A' for eps in quarterly_eps]
        else:
            quarterly_trailing_pe = ["N/A"] * len(available_columns)

        # Format the financial data as a table
        financial_df = pd.DataFrame({
            'Revenue': revenue.apply(format_number),
            'Operating Profit': [f"{format_number(op)} ({r:.2f}%)" if r != 'N/A' else "N/A" for op, r in zip(op_profit, op_profit_ratio)],
            'Net Profit': [f"{format_number(net)} ({r:.2f}%)" if r != 'N/A' else "N/A" for net, r in zip(net_profit, net_profit_ratio)],
            'Trailing P/E': quarterly_trailing_pe,
            'Forward P/E': [forward_pe] + ['N/A'] * (len(available_columns) - 1)
        }, index=available_columns)

        print(f"\nQuarterly Financial Data for {stock_symbol}:")
        print(financial_df.to_string())

        # Display EPS data
        print(f"\nTrailing EPS: {trailing_eps} (Earnings per share for the past 12 months)")
        print(f"Forward EPS: {forward_eps} (Projected earnings per share for the next 12 months)")

        # Fetch institutional holders' data
        institutional_holders = stock.institutional_holders
        if institutional_holders is not None and not institutional_holders.empty:
            if 'pctHeld' in institutional_holders.columns:
                institutional_holders['pctHeld'] = institutional_holders['pctHeld'].apply(lambda x: f"{x * 100:.2f}%" if pd.notnull(x) else 'N/A')

            columns_to_display = [col for col in ['Date Reported', 'Holder', 'Shares', 'Value', 'pctHeld'] if col in institutional_holders.columns]
            print("\nTop 10 Institutional Holders:")
            print(institutional_holders[columns_to_display].head(10).to_string(index=False))
        else:
            print("\nNo institutional holder data available.")

    except Exception as e:
        print(f"An error occurred while fetching data for {stock_symbol}: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python perf.py <symbol>\nExample: python perf.py AAPL")
        sys.exit(1)

    stock_symbol = sys.argv[1].upper()
    get_financial_data(stock_symbol)

