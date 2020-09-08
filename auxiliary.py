import numpy as np

def relu(Z):
    # Rectified Linear Unit Function used for the traversal of the Neural Network
    return np.maximum(0, Z)

def relu_derivative(Z):
    # Derivative of the ReLU function (useful for back propagation)
    Z[Z <= 0] = 0
    Z[Z > 0] = 1
    return Z

def sigmoid(Z):
    # Sigmoid (Logistic) Function used for the last step of the forward propagation phase
    return 1 / (1 + np.exp(-Z))

def sigmoid_derivative(Z):
    # Derivative of the Sigmoid (Logistic) Function (useful for back propagation)
    return sigmoid(Z) * (1 - sigmoid(Z))

def logarithm(Z):
    # Logarithm function that does not allow 0 values (because log(0) = -infinity)
    constant = 0.0000001
    Z = np.where(Z == 0.0, constant, Z)
    return np.log(Z)
