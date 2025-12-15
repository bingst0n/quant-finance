import makepreds
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
predictions = makepreds.predictions

funds = 1000

total_pct = 0

for ticker in predictions.index:
    predictions.loc[ticker, 'predicted_return_pct'] = max(0, predictions.loc[ticker, 'predicted_return_pct'])
    predicted_return = predictions.loc[ticker, 'predicted_return_pct']
    total_pct += predicted_return
    #print(f"{ticker}: {predicted_return}%")
print(f"Total Funds: {funds}")
print(f"Total Percentage: {total_pct}")
print(f"Total Predicted Return: {funds*total_pct/100}")

funds_per_pct = funds/total_pct
print(f"Funds/Pct: {funds_per_pct}")

# Create allocation table
allocation = pd.DataFrame(index=predictions.index, columns=['predicted_return_pct', 'allocation'])

for ticker in predictions.index:
    predicted_return = predictions.loc[ticker, 'predicted_return_pct']
    ticker_allocation = funds_per_pct * predicted_return
    allocation.loc[ticker, 'predicted_return_pct'] = predicted_return
    allocation.loc[ticker, 'allocation'] = ticker_allocation

print("\nPortfolio Allocation:")
print(allocation)
print(f"\nTotal Allocated: ${allocation['allocation'].sum():.2f}")

# Get current prices and calculate shares
tickers = allocation.index.tolist()
prices_data = yf.download(tickers=tickers, period='1d', interval='1d', progress=False)
current_prices = prices_data['Close'].iloc[-1]

# Add shares column to allocation table
allocation['current_price'] = 0.0
allocation['shares'] = 0.0

for ticker in allocation.index:
    if allocation.loc[ticker, 'allocation'] > 0:
        price = current_prices[ticker]
        shares = allocation.loc[ticker, 'allocation'] / price
        allocation.loc[ticker, 'current_price'] = price
        allocation.loc[ticker, 'shares'] = shares

print("\nPortfolio with Shares:")
print(allocation)

# Create bar chart of shares allocation (only non-zero positions)
shares_for_plot = allocation[allocation['shares'] > 0]['shares']

if len(shares_for_plot) > 0:
    plt.figure(figsize=(12, 8))
    plt.bar(shares_for_plot.index, shares_for_plot.values, color='steelblue')
    plt.title('Portfolio Allocation by Shares', fontsize=14, fontweight='bold')
    plt.xlabel('Ticker', fontsize=12)
    plt.ylabel('Number of Shares', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()
else:
    print("\nNo shares to display in bar chart.")