import pandas as pd

# Read weights from CSV into DataFrame
dfba = pd.read_csv('neuralnetwork/model/weights/weights_ba.csv')
dfcb = pd.read_csv('neuralnetwork/model/weights/weights_cb.csv')
dfdc = pd.read_csv('neuralnetwork/model/weights/weights_dc.csv')

learning_rate = 0.01

layer_b = [0, 0, 0, 0, 0, 0, 0, 0]
layer_c = [0, 0, 0, 0]
output = 0

def LeakyReLU(x):
    return max(0.01*x, x)

def forwardPropagate(input):

    # INPUT -> HIDDEN 1

    for i in range(1,len(layer_b)+1):
        neuronsandweights = 0
        for j in range (1, len(input)+1):
            name_to_call = f"Wba{i}{j}"
            neuronsandweights += input[j-1]*dfba[dfba['weight'] == name_to_call]['value'].values[0]
        layer_b[i-1] = LeakyReLU(neuronsandweights)

    print(f"Hidden Layer 1: {layer_b}")

    # HIDDEN 1 -> HIDDEN 2

    for i in range(1,len(layer_c)+1):
        neuronsandweights = 0
        for j in range (1, len(layer_b)+1):
            name_to_call = f"Wcb{i}{j}"
            neuronsandweights += layer_b[j-1]*dfcb[dfcb['weight'] == name_to_call]['value'].values[0]
        layer_c[i-1] = LeakyReLU(neuronsandweights)

    print(f"Hidden Layer 2: {layer_c}")

    # HIDDEN 2 -> OUTPUT

    neuronsandweights = 0
    for j in range (1, len(layer_c)+1):
        name_to_call = f"Wdc1{j}"
        neuronsandweights += layer_c[j-1]*dfdc[dfdc['weight'] == name_to_call]['value'].values[0]
    output = neuronsandweights

    print(f"Final Logit: {output}")
    return output

forwardPropagate([1, 1, 1, 1, 1, 1])

def computeLoss(logit, real):
    return (1/2)*math.pow((real - logit),2)