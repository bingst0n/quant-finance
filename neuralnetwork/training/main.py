import os
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from training import Train, load_trained_weights, get_loss_history, reset_loss_history, reset_gradient_accumulators, apply_accumulated_gradients, save_weights_to_csv
from samples import prepare_samples
from initialization import initialize_weights as init_weights

# ============= TRAINING PARAMETERS =============
TICKER = 'MSFT'
PERIOD = '30y'
BATCH_SIZE = 32
# ===============================================

def prepare_data(ticker, period):
    """Prepare training data using samples.py"""
    print(f"Preparing data for {ticker} with period {period}...")
    data = prepare_samples(ticker, period)
    print(f"Data preparation complete. {len(data)} samples ready.")
    return data

def initialize_weights():
    """Initialize random weights and biases"""
    print("Initializing random weights and biases...")
    init_weights()
    print("Weight initialization complete.")

def train_model(data):
    """Train the model on all rows of data"""
    print(f"\nTraining on {len(data)} rows of data...")
    
    # Separate features and target
    X = data[['lag1', 'lag2', 'vol12', 'mom12', 'drawdown', 'sma']].values
    y = data['real'].values
    
    # STANDARDIZE FEATURES (CRUCIAL FOR GENERALIZATION)
    print("Standardizing features...")
    mean = X.mean(axis=0)  # shape (6,)
    std = X.std(axis=0) + 1e-8  # add small epsilon to avoid division by zero
    X_normalized = (X - mean) / std
    
    print(f"Feature means: {mean}")
    print(f"Feature stds: {std}")
    
    total_rows = len(X_normalized)
    num_batches = (total_rows + BATCH_SIZE - 1) // BATCH_SIZE  # Ceiling division
    
    print(f"Training with batch size {BATCH_SIZE} ({num_batches} batches)")
    
    # Train in mini-batches
    total_loss = 0
    batch_count = 0
    
    for batch_idx in range(num_batches):
        # Reset gradient accumulators for new batch
        reset_gradient_accumulators()
        
        # Get batch boundaries
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, total_rows)
        batch_actual_size = end_idx - start_idx
        
        # Accumulate gradients over batch
        batch_loss = 0
        for i in range(start_idx, end_idx):
            loss = Train(X_normalized[i].tolist(), y[i])
            batch_loss += loss
            total_loss += loss
        
        # Apply accumulated gradients (averaged over batch)
        apply_accumulated_gradients()
        batch_count += 1
        
        # Dynamic progress update
        samples_processed = end_idx
        percentage = (samples_processed / total_rows) * 100
        avg_batch_loss = batch_loss / batch_actual_size
        print(f"\rTRAINING ({percentage:.1f}%, batch {batch_count}/{num_batches}, batch_loss={avg_batch_loss:.6f})", end='', flush=True)

    avg_loss = total_loss / total_rows
    print(f"\n\nTraining complete! Average Loss: {avg_loss:.6f}")
    
    # Return normalization parameters along with loss
    return avg_loss, mean, std

def save_trained_model(ticker, period, mean, std):
    """Save trained weights and biases to models/TICKER_PERIOD"""
    period_years = period.rstrip('y')
    model_name = f"{ticker}{period_years}"
    save_dir = f'neuralnetwork/models/{model_name}'
    
    print(f"\nSaving trained model to {save_dir}...")
    
    # Create directories
    os.makedirs(f'{save_dir}/weights', exist_ok=True)
    os.makedirs(f'{save_dir}/biases', exist_ok=True)
    
    # Copy trained weights
    shutil.copy('neuralnetwork/training/model/trained/weights/weights_ba.csv', f'{save_dir}/weights/weights_ba.csv')
    shutil.copy('neuralnetwork/training/model/trained/weights/weights_cb.csv', f'{save_dir}/weights/weights_cb.csv')
    shutil.copy('neuralnetwork/training/model/trained/weights/weights_dc.csv', f'{save_dir}/weights/weights_dc.csv')
    
    # Copy trained biases
    shutil.copy('neuralnetwork/training/model/trained/biases/biases_b.csv', f'{save_dir}/biases/biases_b.csv')
    shutil.copy('neuralnetwork/training/model/trained/biases/biases_c.csv', f'{save_dir}/biases/biases_c.csv')
    shutil.copy('neuralnetwork/training/model/trained/biases/biases_d.csv', f'{save_dir}/biases/biases_d.csv')
    
    # Save normalization parameters (CRUCIAL FOR INFERENCE)
    normalization_params = pd.DataFrame({
        'feature': ['lag1', 'lag2', 'vol12', 'mom12', 'drawdown', 'sma'],
        'mean': mean,
        'std': std
    })
    normalization_params.to_csv(f'{save_dir}/normalization.csv', index=False)
    print(f"Normalization parameters saved to {save_dir}/normalization.csv")
    
    print(f"Model saved successfully to {save_dir}")

def plot_loss_history(ticker, period):
    """Create and save loss vs iterations plot"""
    print("\nGenerating loss plot...")
    
    loss_df = get_loss_history()
    
    plt.figure(figsize=(10, 6))
    plt.plot(loss_df['iteration'], loss_df['loss'], linewidth=1, alpha=0.7)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(f'Training Loss Over Time - {ticker} ({period})', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    period_years = period.rstrip('y')
    model_name = f"{ticker}{period_years}"
    plot_path = f'neuralnetwork/models/{model_name}/loss_history.png'
    plt.savefig(plot_path, dpi=300)
    print(f"Loss plot saved to {plot_path}")
    
    # Show plot
    #plt.show()

def cleanup():
    """Clear initialization and trained weights and biases"""
    print("\nCleaning up temporary files...")
    
    # Clear initial weights and biases
    initial_dirs = [
        'neuralnetwork/training/model/initial/weights',
        'neuralnetwork/training/model/initial/biases'
    ]
    
    # Clear trained weights and biases
    trained_dirs = [
        'neuralnetwork/training/model/trained/weights',
        'neuralnetwork/training/model/trained/biases'
    ]
    
    for directory in initial_dirs + trained_dirs:
        if os.path.exists(directory):
            shutil.rmtree(directory)
            os.makedirs(directory, exist_ok=True)
    
    print("Cleanup complete.")

def train_ticker(ticker, period, batch_size):
    """Main training pipeline for a given ticker"""
    global BATCH_SIZE
    BATCH_SIZE = batch_size
    
    print("=" * 60)
    print(f"NEURAL NETWORK TRAINING PIPELINE")
    print(f"Ticker: {ticker} | Period: {period} | Batch Size: {batch_size}")
    print("=" * 60)
    
    # Step 1: Prepare data
    data = prepare_data(ticker, period)
    
    # Step 2: Initialize weights
    initialize_weights()
    
    # Step 3: Load weights into training module
    print("Loading weights for training...")
    load_trained_weights()
    reset_loss_history()  # Reset loss tracking for new training run
    print("Weights loaded.")
    
    # Step 4: Train the model
    avg_loss, mean, std = train_model(data)
    
    # Step 4.5: Save weights to CSV
    print("Saving trained weights to CSV...")
    save_weights_to_csv()
    print("Weights saved to CSV.")
    
    # Step 5: Save trained model
    save_trained_model(ticker, period, mean, std)
    
    # Step 6: Plot loss history
    plot_loss_history(ticker, period)
    
    # Step 7: Cleanup
    cleanup()
    
    print("\n" + "=" * 60)
    print("TRAINING PIPELINE COMPLETE!")
    print("=" * 60)

def main():
    """Main training pipeline"""
    train_ticker(TICKER, PERIOD, BATCH_SIZE)

if __name__ == "__main__":
    main()