import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import os
import string
import shutil

ticker = 'AAPL'

# Create model directories if they don't exist
os.makedirs('neuralnetwork/model/biases', exist_ok=True)
os.makedirs('neuralnetwork/model/weights', exist_ok=True)

def heUniformLimit(neurons):
    return [-np.sqrt(6 / neurons), np.sqrt(6 / neurons)]

Wba=heUniformLimit(6)
Wcb=heUniformLimit(8)
Wdc=heUniformLimit(4)

# Process biases files
blank_biases_dir = 'neuralnetwork/blank/biases'
if os.path.exists(blank_biases_dir):
    for filename in os.listdir(blank_biases_dir):
        if filename.endswith('_blank.csv'):
            source_path = os.path.join(blank_biases_dir, filename)
            new_filename = filename.replace('_blank', '')
            dest_path = os.path.join('neuralnetwork/model/biases', new_filename)
            shutil.copy(source_path, dest_path)
            df = pd.read_csv(dest_path)
            df['value'] = 0
            df.to_csv(dest_path, index=False)

# Process weights files
blank_weights_dir = 'neuralnetwork/blank/weights'
if os.path.exists(blank_weights_dir):
    for filename in os.listdir(blank_weights_dir):
        if filename.endswith('_blank.csv'):
            source_path = os.path.join(blank_weights_dir, filename)
            new_filename = filename.replace('_blank', '')
            dest_path = os.path.join('neuralnetwork/model/weights', new_filename)
            shutil.copy(source_path, dest_path)
            if 'ba' in new_filename:
                df = pd.read_csv(dest_path)
                df['value'] = np.random.uniform(Wba[0], Wba[1], size=len(df))
                df.to_csv(dest_path, index=False)
            elif 'cb' in new_filename:
                df = pd.read_csv(dest_path)
                df['value'] = np.random.uniform(Wcb[0], Wcb[1], size=len(df))
                df.to_csv(dest_path, index=False)
            elif 'dc' in new_filename:
                df = pd.read_csv(dest_path)
                df['value'] = np.random.uniform(Wdc[0], Wdc[1], size=len(df))
                df.to_csv(dest_path, index=False)

