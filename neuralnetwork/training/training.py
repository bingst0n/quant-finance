import pandas as pd
import os
import shutil
import math

# Global dataframes for weights and biases
dfba = None
dfcb = None
dfdc = None
dfb_bias = None
dfc_bias = None
dfd_bias = None

# Loss tracking
loss_history = pd.DataFrame(columns=['iteration', 'loss'])
iteration_counter = 0

# Gradient accumulators for mini-batch training
gradient_accumulators = {
    'wdc': {},
    'wcb': {},
    'wba': {},
    'bd': 0,
    'bc': {},
    'bb': {}
}
batch_sample_count = 0

def load_trained_weights():
    """Load initial weights/biases into trained directory and read them"""
    global dfba, dfcb, dfdc, dfb_bias, dfc_bias, dfd_bias
    
    # Create trained directories
    os.makedirs('neuralnetwork/training/model/trained/weights', exist_ok=True)
    os.makedirs('neuralnetwork/training/model/trained/biases', exist_ok=True)
    
    # Copy initial weights to trained directory
    shutil.copy('neuralnetwork/training/model/initial/weights/weights_ba.csv', 'neuralnetwork/training/model/trained/weights/weights_ba.csv')
    shutil.copy('neuralnetwork/training/model/initial/weights/weights_cb.csv', 'neuralnetwork/training/model/trained/weights/weights_cb.csv')
    shutil.copy('neuralnetwork/training/model/initial/weights/weights_dc.csv', 'neuralnetwork/training/model/trained/weights/weights_dc.csv')
    
    # Copy initial biases to trained directory
    shutil.copy('neuralnetwork/training/model/initial/biases/biases_b.csv', 'neuralnetwork/training/model/trained/biases/biases_b.csv')
    shutil.copy('neuralnetwork/training/model/initial/biases/biases_c.csv', 'neuralnetwork/training/model/trained/biases/biases_c.csv')
    shutil.copy('neuralnetwork/training/model/initial/biases/biases_d.csv', 'neuralnetwork/training/model/trained/biases/biases_d.csv')
    
    # Read weights from CSV into DataFrame (from trained directory)
    dfba = pd.read_csv('neuralnetwork/training/model/trained/weights/weights_ba.csv')
    dfcb = pd.read_csv('neuralnetwork/training/model/trained/weights/weights_cb.csv')
    dfdc = pd.read_csv('neuralnetwork/training/model/trained/weights/weights_dc.csv')
    
    # Read biases from CSV into DataFrame (from trained directory)
    dfb_bias = pd.read_csv('neuralnetwork/training/model/trained/biases/biases_b.csv')
    dfc_bias = pd.read_csv('neuralnetwork/training/model/trained/biases/biases_c.csv')
    dfd_bias = pd.read_csv('neuralnetwork/training/model/trained/biases/biases_d.csv')

learning_rate = 0.005

layer_b = [0, 0, 0, 0, 0, 0, 0, 0]
layer_c = [0, 0, 0, 0]
pre_activation_b = [0, 0, 0, 0, 0, 0, 0, 0]
pre_activation_c = [0, 0, 0, 0]
output = 0

def LeakyReLU(x):
    return max(0.01*x, x)

def DerivativeLeakyReLU(x):
    if x < 0:
        return 0.01
    else:
        return 1

def forwardPropagate(input):
    # INPUT -> HIDDEN 1
    for i in range(1,len(layer_b)+1):
        neuronsandweights = 0
        for j in range (1, len(input)+1):
            name_to_call = f"Wba{i}{j}"
            neuronsandweights += input[j-1]*dfba[dfba['weight'] == name_to_call]['value'].values[0]
        # Add bias
        bias_name = f"Bb{i}"
        bias = dfb_bias[dfb_bias['bias'] == bias_name]['value'].values[0]
        neuronsandweights += bias
        pre_activation_b[i-1] = neuronsandweights
        layer_b[i-1] = LeakyReLU(neuronsandweights)
    # HIDDEN 1 -> HIDDEN 2
    for i in range(1,len(layer_c)+1):
        neuronsandweights = 0
        for j in range (1, len(layer_b)+1):
            name_to_call = f"Wcb{i}{j}"
            neuronsandweights += layer_b[j-1]*dfcb[dfcb['weight'] == name_to_call]['value'].values[0]
        # Add bias
        bias_name = f"Bc{i}"
        bias = dfc_bias[dfc_bias['bias'] == bias_name]['value'].values[0]
        neuronsandweights += bias
        pre_activation_c[i-1] = neuronsandweights
        layer_c[i-1] = LeakyReLU(neuronsandweights)
    # HIDDEN 2 -> OUTPUT
    neuronsandweights = 0
    for j in range (1, len(layer_c)+1):
        name_to_call = f"Wdc1{j}"
        neuronsandweights += layer_c[j-1]*dfdc[dfdc['weight'] == name_to_call]['value'].values[0]
    # Add output bias
    bias = dfd_bias[dfd_bias['bias'] == 'Bd1']['value'].values[0]
    neuronsandweights += bias
    output = neuronsandweights
    return output

def computeLoss(logit, real):
    global loss_history, iteration_counter
    loss = (1/2)*math.pow((real - logit),2)
    
    # Track loss in history
    loss_history.loc[len(loss_history)] = [iteration_counter, loss]
    iteration_counter += 1
    
    return loss

def get_loss_history():
    """Return the loss history DataFrame"""
    return loss_history.copy()

def reset_loss_history():
    """Reset loss history for new training run"""
    global loss_history, iteration_counter
    loss_history = pd.DataFrame(columns=['iteration', 'loss'])
    iteration_counter = 0

def reset_gradient_accumulators():
    """Reset gradient accumulators to zero for a new batch"""
    global gradient_accumulators, batch_sample_count
    gradient_accumulators = {
        'wdc': {},
        'wcb': {},
        'wba': {},
        'bd': 0,
        'bc': {},
        'bb': {}
    }
    batch_sample_count = 0

def accumulate_gradients(logit, real, input):
    """Compute and accumulate gradients for one sample (no weight updates)"""
    global gradient_accumulators, batch_sample_count
    
    # NEURON ERRORS
    neuron_errors_d = [(logit - real)]
    neuron_errors_c = []
    for i in range(1, len(layer_c)+1):
        name_to_call = f"Wdc1{i}"
        weight_dc1i = dfdc[dfdc['weight'] == name_to_call]['value'].values[0]
        neuron_errors_c.append(weight_dc1i * neuron_errors_d[0] * DerivativeLeakyReLU(pre_activation_c[i-1]))
    neuron_errors_b = []
    for i in range(1, len(layer_b)+1):
        error_sum = 0
        for j in range (1, len(layer_c)+1):
            name_to_call = f"Wcb{j}{i}"
            weight_cbji = dfcb[dfcb['weight'] == name_to_call]['value'].values[0]
            error_sum += neuron_errors_c[j-1] * weight_cbji
        neuron_errors_b.append(error_sum * DerivativeLeakyReLU(pre_activation_b[i-1]))
    
    # ACCUMULATE GRADIENTS (no updates yet)
    # Hidden Layer 2 -> Output (dfdc)
    for i in range(1, len(layer_c)+1):
        name_to_call = f"Wdc1{i}"
        gradient = neuron_errors_d[0] * layer_c[i-1]
        if name_to_call not in gradient_accumulators['wdc']:
            gradient_accumulators['wdc'][name_to_call] = 0
        gradient_accumulators['wdc'][name_to_call] += gradient
    
    # Hidden Layer 1 -> Hidden Layer 2 (dfcb)
    for i in range(1, len(layer_c)+1):
        for j in range(1, len(layer_b)+1):
            name_to_call = f"Wcb{i}{j}"
            gradient = neuron_errors_c[i-1] * layer_b[j-1]
            if name_to_call not in gradient_accumulators['wcb']:
                gradient_accumulators['wcb'][name_to_call] = 0
            gradient_accumulators['wcb'][name_to_call] += gradient
    
    # Input -> Hidden Layer 1 (dfba)
    for i in range(1, len(layer_b)+1):
        for j in range(1, len(input)+1):
            name_to_call = f"Wba{i}{j}"
            gradient = neuron_errors_b[i-1] * input[j-1]
            if name_to_call not in gradient_accumulators['wba']:
                gradient_accumulators['wba'][name_to_call] = 0
            gradient_accumulators['wba'][name_to_call] += gradient

    # ACCUMULATE BIAS GRADIENTS
    # Output bias
    gradient_accumulators['bd'] += neuron_errors_d[0]
    
    # Hidden Layer 2 biases
    for i in range(1, len(layer_c)+1):
        bias_name = f"Bc{i}"
        if bias_name not in gradient_accumulators['bc']:
            gradient_accumulators['bc'][bias_name] = 0
        gradient_accumulators['bc'][bias_name] += neuron_errors_c[i-1]
    
    # Hidden Layer 1 biases
    for i in range(1, len(layer_b)+1):
        bias_name = f"Bb{i}"
        if bias_name not in gradient_accumulators['bb']:
            gradient_accumulators['bb'][bias_name] = 0
        gradient_accumulators['bb'][bias_name] += neuron_errors_b[i-1]
    
    batch_sample_count += 1

def apply_accumulated_gradients():
    """Apply accumulated gradients (averaged over batch) to update weights"""
    global batch_sample_count
    
    if batch_sample_count == 0:
        return
    
    # Average gradients over batch
    batch_size = batch_sample_count
    
    # Update weights - Hidden Layer 2 -> Output (dfdc)
    for name_to_call, gradient_sum in gradient_accumulators['wdc'].items():
        current_weight = dfdc[dfdc['weight'] == name_to_call]['value'].values[0]
        avg_gradient = gradient_sum / batch_size
        new_weight = current_weight - learning_rate * avg_gradient
        dfdc.loc[dfdc['weight'] == name_to_call, 'value'] = new_weight
    
    # Update weights - Hidden Layer 1 -> Hidden Layer 2 (dfcb)
    for name_to_call, gradient_sum in gradient_accumulators['wcb'].items():
        current_weight = dfcb[dfcb['weight'] == name_to_call]['value'].values[0]
        avg_gradient = gradient_sum / batch_size
        new_weight = current_weight - learning_rate * avg_gradient
        dfcb.loc[dfcb['weight'] == name_to_call, 'value'] = new_weight
    
    # Update weights - Input -> Hidden Layer 1 (dfba)
    for name_to_call, gradient_sum in gradient_accumulators['wba'].items():
        current_weight = dfba[dfba['weight'] == name_to_call]['value'].values[0]
        avg_gradient = gradient_sum / batch_size
        new_weight = current_weight - learning_rate * avg_gradient
        dfba.loc[dfba['weight'] == name_to_call, 'value'] = new_weight
    
    # Update biases - Output
    current_bias = dfd_bias[dfd_bias['bias'] == 'Bd1']['value'].values[0]
    avg_gradient = gradient_accumulators['bd'] / batch_size
    new_bias = current_bias - learning_rate * avg_gradient
    dfd_bias.loc[dfd_bias['bias'] == 'Bd1', 'value'] = new_bias
    
    # Update biases - Hidden Layer 2
    for bias_name, gradient_sum in gradient_accumulators['bc'].items():
        current_bias = dfc_bias[dfc_bias['bias'] == bias_name]['value'].values[0]
        avg_gradient = gradient_sum / batch_size
        new_bias = current_bias - learning_rate * avg_gradient
        dfc_bias.loc[dfc_bias['bias'] == bias_name, 'value'] = new_bias
    
    # Update biases - Hidden Layer 1
    for bias_name, gradient_sum in gradient_accumulators['bb'].items():
        current_bias = dfb_bias[dfb_bias['bias'] == bias_name]['value'].values[0]
        avg_gradient = gradient_sum / batch_size
        new_bias = current_bias - learning_rate * avg_gradient
        dfb_bias.loc[dfb_bias['bias'] == bias_name, 'value'] = new_bias

def save_weights_to_csv():
    """Save updated weights and biases from memory to CSV files"""
    # Save weights
    dfba.to_csv('neuralnetwork/training/model/trained/weights/weights_ba.csv', index=False)
    dfcb.to_csv('neuralnetwork/training/model/trained/weights/weights_cb.csv', index=False)
    dfdc.to_csv('neuralnetwork/training/model/trained/weights/weights_dc.csv', index=False)
    
    # Save biases
    dfb_bias.to_csv('neuralnetwork/training/model/trained/biases/biases_b.csv', index=False)
    dfc_bias.to_csv('neuralnetwork/training/model/trained/biases/biases_c.csv', index=False)
    dfd_bias.to_csv('neuralnetwork/training/model/trained/biases/biases_d.csv', index=False)

def Train(input_data, target):
    """Train on a single sample - accumulates gradients without updating weights"""
    # Forward propagation
    logit = forwardPropagate(input_data)
    
    # Compute loss
    loss = computeLoss(logit, target)
    
    # Accumulate gradients (no weight updates)
    accumulate_gradients(logit, target, input_data)
    
    return loss