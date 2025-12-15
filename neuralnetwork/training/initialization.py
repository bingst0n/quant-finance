import numpy as np
import pandas as pd
import os
import shutil

def heUniformLimit(neurons):
    return [-np.sqrt(6 / neurons), np.sqrt(6 / neurons)]

def initialize_weights():
    """Initialize random weights and biases from blank templates"""
    # Create model directories if they don't exist
    os.makedirs('neuralnetwork/training/model/initial/biases', exist_ok=True)
    os.makedirs('neuralnetwork/training/model/initial/weights', exist_ok=True)
    
    # Calculate initialization limits
    Wba = heUniformLimit(6)
    Wcb = heUniformLimit(8)
    Wdc = heUniformLimit(4)
    
    # Bias initialization limits (same as corresponding layer weights)
    Bb = heUniformLimit(6)
    Bc = heUniformLimit(8)
    Bd = heUniformLimit(4)
    
    # Process biases files
    blank_biases_dir = 'neuralnetwork/training/blank/biases'
    if os.path.exists(blank_biases_dir):
        for filename in os.listdir(blank_biases_dir):
            if filename.endswith('_blank.csv'):
                source_path = os.path.join(blank_biases_dir, filename)
                new_filename = filename.replace('_blank', '')
                dest_path = os.path.join('neuralnetwork/training/model/initial/biases', new_filename)
                shutil.copy(source_path, dest_path)
                df = pd.read_csv(dest_path)
                
                # Initialize biases with random values based on layer
                if 'b' in new_filename:  # Hidden layer 1
                    df['value'] = np.random.uniform(Bb[0], Bb[1], size=len(df))
                elif 'c' in new_filename:  # Hidden layer 2
                    df['value'] = np.random.uniform(Bc[0], Bc[1], size=len(df))
                elif 'd' in new_filename:  # Output layer
                    df['value'] = np.random.uniform(Bd[0], Bd[1], size=len(df))
                
                df.to_csv(dest_path, index=False)
    
    # Process weights files
    blank_weights_dir = 'neuralnetwork/training/blank/weights'
    if os.path.exists(blank_weights_dir):
        for filename in os.listdir(blank_weights_dir):
            if filename.endswith('_blank.csv'):
                source_path = os.path.join(blank_weights_dir, filename)
                new_filename = filename.replace('_blank', '')
                dest_path = os.path.join('neuralnetwork/training/model/initial/weights', new_filename)
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

