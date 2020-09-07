# GRADED FUNCTION: two_layer_model
import numpy as np
from auxiliary import *

def model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):
    """
    Implements a three-layer artificial neural network: LINEAR->RELU->LINEAR->SIGMOID.

    Arguments:
    X -- input data, of shape (n_x, number of examples)
    Y -- vector of elements from 0 to 23? (corresponding to each letter of the English Alphabet)
    layers_dims -- dimensions of the layers (n_x, n_h, n_y)
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- If set to True, this will print the cost every 100 iterations

    Returns:
    parameters -- a dictionary containing W1, W2, b1, and b2
    """

    np.random.seed(1)  # initialise a random seed
    grads = {}  # dictionary of gradients
    costs = []  # to keep track of the cost
    m = X.shape[1]  # number of examples
    (n_x, n_h, n_y) = layers_dims  # unboxing the layer dimensions

    # Initialise the parameters randomly
    parameters = {}
    dimensions = [n_x] + n_h + [n_y]
    for l in range(1, len(dimensions)):
        parameters['W'+str(l)] = np.random.randn(dimensions[l], dimensions[l-1]) * 0.01
        parameters['b' + str(l)] = np.zeros((dimensions[l], 1))

    # Start Optimization - Batch Gradient Descent
    activations = {}  # Dictionary to hold the activation vectors ( A_0, A_1, ... A_(L-1) )
    cache = {}  # Dictionary in which we cache the results of Z (we need them for back propagation algorithm)
    for i in range(0, num_iterations):

        # Forward Propagation Step
        # FIX SIZES HERE
        activations['A' + str(0)] = X
        for j in range(1, len(dimensions)):
            A_prev = activations['A' + str(j-1)]  # Take the previous activation vector (initially the input vector)
            W = parameters['W' + str(j)]  # Retrieve the weights from the parameters dictionary
            b = parameters['b' + str(j)]  # Retrieve the bias from the parameters dictionary
            Z = np.dot(W, A_prev) + b  # Compute the Z value for the current layer
            if j == len(dimensions) - 1:  # Compute the activation for the current layer
                A = sigmoid(Z)
            else:
                A = relu(Z)
            activations['A' + str(j)] = A  # Store the activation so we can use it for the following iteration
            cache['Z' + str(j)] = Z  # Cache the current Z vector to be used later in the back propagation phase
            print(A.shape)

        # Cost computation
        A_last = activations['A' + str(len(dimensions) - 1)]  # Taking the last activation vector
        print(A_last.shape)
        cost = 1 / m * np.sum(Y*np.log(A_last) + (1-Y)*np.log(1-A_last))
        # Backward Propagation Step

        # Updating parameters (weights and biases)

        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if print_cost and i % 100 == 0:
            costs.append(cost)
        '''

    # plot the cost

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters
    
'''