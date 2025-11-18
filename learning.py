# Interval of MONTHS used throughout

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from functions import *

ticker1 = 'MSFT'

data = yf.download(tickers=[ticker1], period='5y', interval='1mo')
prices = data['Close'].copy()
prices.dropna(inplace=True)
prices = prices.to_period('M')

rets = prices.pct_change().dropna()

wealthIndex = (1+rets).cumprod()
start_date = wealthIndex.index.min() - 1
wealthIndex = pd.concat([pd.DataFrame([[1]], index=[start_date], columns=wealthIndex.columns), wealthIndex])

previousPeaks = wealthIndex.cummax()
drawdowns = (wealthIndex - previousPeaks) / previousPeaks

sharpeRatio = annualizeRets(rets, ppy=52) / annualizeVol(rets, ppy=52)

print(rets)

plt.xlabel("Time")
plt.plot(wealthIndex.to_timestamp(), label='Wealth Index')
plt.plot(previousPeaks.to_timestamp(), label='Previous Peaks')
plt.plot(drawdowns.to_timestamp(), label='Drawdowns')
plt.legend()
plt.show()