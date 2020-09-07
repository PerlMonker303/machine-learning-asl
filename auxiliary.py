import numpy as np

def relu(Z):
    # Rectified Linear Unit Function used for the traversal of the Neural Network
    return np.maximum(0, Z)

def sigmoid(Z):
    # Sigmoid (logistic) Function used for the last step of the forward propagation phase
    return 1 / 1 + np.exp(-Z)