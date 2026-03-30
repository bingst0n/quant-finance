import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import os

def prepare_samples(ticker, period):
    # Download data
    data = yf.download(tickers=[ticker], period=period, interval='1wk')
    prices = data['Close'].copy()
    prices.dropna(inplace=True)
    prices = prices.to_period('W')
    
    # Calculate returns
    rets = prices.pct_change().dropna()
    
    # Create information DataFrame
    information = pd.DataFrame(index=rets.index, columns=['real', 'lag1', 'lag2', 'vol12', 'mom12', 'drawdown', 'sma'])
    information['real'] = rets.shift(-1)
    information['lag1'] = rets.shift(1)
    information['lag2'] = information['lag1'].shift(1)
    information['vol12'] = rets.shift(1).rolling(12).std()
    information['mom12'] = prices / prices.shift(12) - 1
    
    # Calculate drawdown
    wealth = (1 + rets).cumprod()
    previous_peaks = wealth.cummax()
    information['drawdown'] = (wealth - previous_peaks) / previous_peaks
    
    # Calculate SMA indicator
    sma4 = prices.rolling(4).mean()
    sma26 = prices.rolling(26).mean()
    information['sma'] = (sma4 / sma26) - 1
    
    # Clean up
    information = information.dropna()
    information.index = information.index.to_timestamp()
    information = information.iloc[:-1]
    
    return information