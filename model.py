import matplotlib.pyplot as plt
from auxiliary import *
from gradientChecking import checkGradient

def model(X, Y, layers_dims, learning_rate=0.075, num_iterations=3000, print_cost=False, lambd=0.5):
    """
    Implements a three-layer artificial neural network

    Arguments:
    X -- input data, of shape (n_x, number of examples)
    Y -- vector of elements from 0 to 24 (corresponding to each letter of the English Alphabet)
    layers_dims -- dimensions of the layers (n_x, n_h, n_y)
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if set to True, this will print the cost every 100 iterations
    lambd -- the regularization factor

    Returns:
    parameters -- a dictionary containing W1, W2, b1, and b2
    """

    costs = []  # to keep track of the cost
    m = X.shape[1]  # number of examples
    (n_x, n_h, n_y) = layers_dims  # unboxing the layer dimensions

    # Initialise the parameters randomly
    parameters = {}
    dimensions = [n_x] + n_h + [n_y]
    for l in range(1, len(dimensions)):  # Using He Initialization
        parameters['W' + str(l)] = np.random.randn(dimensions[l], dimensions[l-1]) * np.sqrt(2 / dimensions[l-1])
        parameters['b' + str(l)] = np.zeros((dimensions[l], 1))

    # Start Optimization - Batch Gradient Descent
    activations = {}  # Dictionary to hold the activation vectors ( A_0, A_1, ... A_(L-1) )
    cache = {}  # Dictionary in which we cache the results of Z (we need them for back propagation algorithm)
    for i in range(0, num_iterations):
        # Forward Propagation Step
        activations['A' + str(0)] = X
        for j in range(1, len(dimensions)):
            A_prev = activations['A' + str(j-1)]  # Take the previous activation vector (initially the input vector)
            W = parameters['W' + str(j)]  # Retrieve the weights from the parameters dictionary
            b = parameters['b' + str(j)]  # Retrieve the bias from the parameters dictionary
            Z = np.dot(W, A_prev) + b  # Compute the Z value for the current layer
            if j == len(dimensions) - 1:  # Compute the activation for the current layer
                A = sigmoid(Z)
            else:
                A = leaky_relu(Z)
            activations['A' + str(j)] = A  # Store the activation so we can use it for the following iteration
            cache['Z' + str(j)] = Z  # Cache the current Z vector to be used later in the back propagation phase

        # Cost computation + Regularization
        A_last = A  # Taking the last activation vector
        cost = - 1 / m * np.sum(np.dot(Y.T, logarithm(A_last)) + np.dot((1-Y).T, logarithm(1-A_last)))
        regularization = 0  # Initialising the regularization
        for l in range(1, len(dimensions)):
            W = parameters['W' + str(l)]
            regularization += np.sum(np.square(W))

        cost += (lambd / m * regularization)
        costs.append(cost)

        # Backward Propagation Step
        grads = dict()  # Declaring an empty gradient dictionary
        grads['dA' + str(len(dimensions) - 1)] = - np.divide(Y, A_last) + np.divide(1 - Y, 1 - A_last)
        for j in range(len(dimensions) - 1, 0, -1):
            dA = grads['dA' + str(j)]
            if j == len(dimensions) - 1:
                dZ = dA * sigmoid_derivative(cache['Z' + str(j)])
            else:
                dZ = dA * relu_derivative(cache['Z' + str(j)])
            A_prev = activations['A' + str(j-1)]
            W = parameters['W' + str(j)]
            dW = 1 / m * np.dot(dZ, A_prev.T) + lambd / m * W
            db = 1 / m * np.sum(dZ, axis=1, keepdims=True)
            grads["dW" + str(j)] = dW
            grads["db" + str(j)] = db
            dA_prev = np.dot(W.T, dZ)
            grads['dA' + str(j-1)] = dA_prev

        # Updating parameters (weights and biases)
        for l in range(len(dimensions)-1):
            parameters["W" + str(l + 1)] -= learning_rate * grads["dW" + str(l + 1)]
            parameters["b" + str(l + 1)] -= learning_rate * grads["db" + str(l + 1)]

        # Print the cost after every iteration
        if print_cost:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))

        if i == 0:
            # Check if the back propagation algorithm functions fine (using Gradient Checking)
            #checkGradient(parameters, grads, X, Y, dimensions, lambd)  # TO BE RAN ONLY ONCE FOR CORRECTITUDE CHECKING
            pass

    # Plot the cost graph at the end

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('# iterations')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters
