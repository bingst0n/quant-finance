import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

ticker1 = 'MSFT'

data = yf.download(tickers=[ticker1], period='30y', interval='1wk')
prices = data['Close'].copy()
prices.dropna(inplace=True)
prices = prices.to_period('W')

rets = prices.pct_change().dropna()
information = pd.DataFrame(index=rets.index, columns=['real', 'lag1', 'lag2', 'vol12', 'mom12', 'drawdown', 'sma'])
information['real'] = rets.shift(0)
information['lag1'] = rets.shift(1)
information['lag2'] = information['lag1'].shift(1)
information['vol12'] = rets.rolling(12).std()
information['mom12'] = prices / prices.shift(12) - 1
wealth = (1 + rets).cumprod()
previous_peaks = wealth.cummax()
information['drawdown'] = (wealth - previous_peaks) / previous_peaks
sma4 = prices.rolling(4).mean()
sma26 = prices.rolling(26).mean()
information['sma'] = (sma4 / sma26) - 1
information = information.dropna()
information.index = information.index.to_timestamp()
information = information.iloc[:-1]

print(information)