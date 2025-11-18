import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import math

def formalizePercentage(num):
    formal = ((num * 100).round(2).astype(str) + "%")
    return formal

def date_to_index(date_input, prices): # Convert date strings like '2020-12' to pandas syntax
    # If already an integer, return as-is
    if isinstance(date_input, int):
        return date_input
    # If string, convert to integer index
    period = pd.Period(date_input, freq='M')
    return prices.index.get_loc(period)

def singlePeriodReturn(ticker, start, end, log, prices):
    start = date_to_index(start, prices)
    end = date_to_index(end, prices)
    stockReturn = (prices[ticker].iloc[end] / prices[ticker].iloc[start]) - 1
    if log:
        return np.log(1 + stockReturn)
    else:
        return stockReturn

def multiPeriodReturn(pr1, pr2, geometric):
    if geometric:
        multiReturn = (1 + pr1)*(1 + pr2) - 1
    else:
        multiReturn = pr1 + pr2
    return multiReturn

def compoundRets(dataset):
    return (dataset + 1).prod() - 1

def annualizeVol(dataset, ppy=12):
    return dataset.std() * ppy**0.5

def annualizeRets(dataset, ppy=12):
    numberPeriods = dataset.shape[0]
    compoundGrowth = compoundRets(dataset)
    return compoundGrowth**(ppy/numberPeriods) - 1

def annualizeVol(dataset, ppy=12):
    return dataset.std() * ppy**0.5