import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

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

data = yf.download(tickers=tickers, period='30y', interval='1wk')
prices = data['Close'].copy()
prices.dropna(inplace=True)

rets = prices.pct_change()

# Create features DataFrame for each ticker
features = pd.DataFrame(index=tickers, columns=['lag1', 'lag2', 'vol12', 'mom12', 'drawdown', 'sma'])

for ticker in tickers:
    ticker_prices = prices[ticker]
    ticker_rets = rets[ticker]
    
    # Get the most recent values
    features.loc[ticker, 'lag1'] = ticker_rets.iloc[-1]
    features.loc[ticker, 'lag2'] = ticker_rets.iloc[-2]
    features.loc[ticker, 'vol12'] = ticker_rets.rolling(12).std().iloc[-1]
    features.loc[ticker, 'mom12'] = (ticker_prices.iloc[-1] / ticker_prices.iloc[-13] - 1) if len(ticker_prices) >= 13 else np.nan
    
    # Calculate drawdown
    wealth = (1 + ticker_rets).cumprod()
    previous_peaks = wealth.cummax()
    drawdown = (wealth - previous_peaks) / previous_peaks
    features.loc[ticker, 'drawdown'] = drawdown.iloc[-1]
    
    # Calculate SMA indicator
    sma4 = ticker_prices.rolling(4).mean()
    sma26 = ticker_prices.rolling(26).mean()
    sma_indicator = (sma4 / sma26) - 1
    features.loc[ticker, 'sma'] = sma_indicator.iloc[-1]

print("\nRaw Features (before normalization):")
print(features)
print("\n" + "="*80 + "\n")

# Normalize features and make predictions
def LeakyReLU(x):
    return max(0.01*x, x)

predictions = pd.DataFrame(index=tickers, columns=['predicted_return_pct'])

for ticker in tickers:
    # Load model files for this ticker
    model_path = f'neuralnetwork/models/{ticker}30'
    
    # Load normalization parameters
    norm_df = pd.read_csv(f'{model_path}/normalization.csv')
    norm_dict = {row['feature']: {'mean': row['mean'], 'std': row['std']} 
                 for _, row in norm_df.iterrows()}
    
    # Load weights
    dfba = pd.read_csv(f'{model_path}/weights/weights_ba.csv')
    dfcb = pd.read_csv(f'{model_path}/weights/weights_cb.csv')
    dfdc = pd.read_csv(f'{model_path}/weights/weights_dc.csv')
    
    # Load biases
    dfb_bias = pd.read_csv(f'{model_path}/biases/biases_b.csv')
    dfc_bias = pd.read_csv(f'{model_path}/biases/biases_c.csv')
    dfd_bias = pd.read_csv(f'{model_path}/biases/biases_d.csv')
    
    # Normalize input features
    input_features = []
    for feature in ['lag1', 'lag2', 'vol12', 'mom12', 'drawdown', 'sma']:
        raw_value = features.loc[ticker, feature]
        normalized_value = (raw_value - norm_dict[feature]['mean']) / norm_dict[feature]['std']
        input_features.append(normalized_value)
    
    # Debug: Print for GLD
    if ticker == 'GLD':
        print(f"GLD Normalization Debug:")
        print(f"Normalization parameters: {norm_dict}")
        print(f"Raw features: {[features.loc[ticker, f] for f in ['lag1', 'lag2', 'vol12', 'mom12', 'drawdown', 'sma']]}")
        print(f"Normalized features: {input_features}")
        print()
    
    # Forward propagation (exact same structure as training.py)
    layer_b_size = 8
    layer_c_size = 4
    
    # INPUT -> HIDDEN LAYER 1 (layer_b)
    layer_b = []
    for i in range(1, layer_b_size + 1):
        neuronsandweights = 0
        for j in range(1, len(input_features) + 1):
            name_to_call = f"Wba{i}{j}"
            neuronsandweights += input_features[j-1] * dfba[dfba['weight'] == name_to_call]['value'].values[0]
        # Add bias
        bias_name = f"Bb{i}"
        bias = dfb_bias[dfb_bias['bias'] == bias_name]['value'].values[0]
        neuronsandweights += bias
        layer_b.append(LeakyReLU(neuronsandweights))
    
    # HIDDEN LAYER 1 -> HIDDEN LAYER 2 (layer_c)
    layer_c = []
    for i in range(1, layer_c_size + 1):
        neuronsandweights = 0
        for j in range(1, layer_b_size + 1):
            name_to_call = f"Wcb{i}{j}"
            neuronsandweights += layer_b[j-1] * dfcb[dfcb['weight'] == name_to_call]['value'].values[0]
        # Add bias
        bias_name = f"Bc{i}"
        bias = dfc_bias[dfc_bias['bias'] == bias_name]['value'].values[0]
        neuronsandweights += bias
        layer_c.append(LeakyReLU(neuronsandweights))
    
    # HIDDEN LAYER 2 -> OUTPUT
    neuronsandweights = 0
    for j in range(1, layer_c_size + 1):
        name_to_call = f"Wdc1{j}"
        neuronsandweights += layer_c[j-1] * dfdc[dfdc['weight'] == name_to_call]['value'].values[0]
    # Add output bias
    bias = dfd_bias[dfd_bias['bias'] == 'Bd1']['value'].values[0]
    neuronsandweights += bias
    output = neuronsandweights
    
    # Store prediction as percentage
    predictions.loc[ticker, 'predicted_return_pct'] = output * 100

predictions.plot(kind='bar')
#plt.show()

print("\nPredictions:")
print(predictions)