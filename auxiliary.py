import numpy as np

def relu(Z):
    # Rectified Linear Unit Function used for the traversal of the Neural Network
    return np.maximum(0, Z)

def relu_derivative(Z):
    # Derivative of the ReLU function (useful for back propagation)
    Z = np.where(Z <= 0, 0, Z)
    Z = np.where(Z > 0, 1, Z)
    return Z

def leaky_relu(Z):
    # Leaky Rectified Linear Unit Function
    c = 0.001
    Z = np.where(Z <= 0, Z * c, Z)
    Z = np.where(Z > 0, 1, Z)
    return Z

def sigmoid(Z):
    # Sigmoid (Logistic) Function used for the last step of the forward propagation phase
    return 1 / (1 + np.exp(-Z))

def sigmoid_derivative(Z):
    # Derivative of the Sigmoid (Logistic) Function (useful for back propagation)
    return sigmoid(Z) * (1 - sigmoid(Z))

def logarithm(Z):
    # Logarithm function that does not allow 0 values (because log(0) = -infinity)
    constant = 0.000001
    Z = np.where(Z == 0.0, constant, Z)
    return np.log(Z)

def tanh(Z):
    # Hyperbolic Tangent function - alternative for the Sigmoid Function
    return (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))

def tanh_derivative(Z):
    # Hyperbolic Tangent Derivative
    return 1 / np.power(np.cosh(Z), 2)

def softmax(Z):
    # Softmax Function
    return np.exp(Z) / np.sum(np.exp(Z), axis=0)

def dictionary_to_vector(dict):
    # Function that given a dictionary will output a vector with the values from the dictionary
    vec = []
    for i in dict.keys():
        vec.append(dict[i])

    return vec

def vector_to_dictionary(vec):
    # Function that given a vector will output a dictionary
    dic = {}
    for i in range(0, len(vec)-1, 2):
        dic['W' + str(int(i/2) + 1)] = vec[i]
        dic['b' + str(int(i/2) + 1)] = vec[i+1]
    return dic

def initialize_adam(parameters, dimensions):
    # Function used to initialise the parameters needed for Adam optimization
    v = {}  # Initialise empty dictionary for v values
    s = {}  # Initialise empty dictionary for s values

    for l in range(len(dimensions) - 1):
        v["dW" + str(l + 1)] = np.zeros((parameters["W" + str(l + 1)].shape[0], parameters["W" + str(l + 1)].shape[1]))
        v["db" + str(l + 1)] = np.zeros((parameters["b" + str(l + 1)].shape[0], parameters["b" + str(l + 1)].shape[1]))
        s["dW" + str(l + 1)] = np.zeros((parameters["W" + str(l + 1)].shape[0], parameters["W" + str(l + 1)].shape[1]))
        s["db" + str(l + 1)] = np.zeros((parameters["b" + str(l + 1)].shape[0], parameters["b" + str(l + 1)].shape[1]))

    return v, s