import sys
sys.path.insert(0, 'neuralnetwork/training')
from main import train_ticker

tickers = [
    "SPY",  # US Large Cap Equity
    "QQQ",  # US Tech Equity
    "IWM",  # US Small Cap Equity
    "EFA",  # International Developed Markets
    "EEM",  # Emerging Markets
    "TLT",  # Long-Term Treasuries
    "IEF",  # Intermediate Treasuries
    "LQD",  # Investment-Grade Corporate Bonds
    "HYG",  # High-Yield Corporate Bonds
    "GLD",  # Gold
]

PERIOD = '30y'
BATCH_SIZE = 32

for ticker in tickers:
    train_ticker(ticker, PERIOD, BATCH_SIZE)