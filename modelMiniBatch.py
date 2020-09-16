import matplotlib.pyplot as plt
from auxiliary import *
import math

def modelMiniBatch(X, Y, layers_dims, mini_batch_size, learning_rate=0.075, decay_rate = 1, num_epochs=1000, print_cost=False, lambd=0.5):
    """
    Implements a three-layer artificial neural network
    (optimization using Mini-Batch Gradient Descent and Adam Optimizer)

    Arguments:
    X -- input data, of shape (n_x, number of examples)
    Y -- vector of elements from 0 to 24 (corresponding to each letter of the English Alphabet)
    layers_dims -- dimensions of the layers (n_x, n_h, n_y)
    mini_batch_size -- size of a mini batch (1 => Stochastic G.D., m => Batch G. D.)
    learning_rate -- learning rate of the gradient descent update rule
    decay_rate -- constant used for decaying the learning rate for each epoch
    num_epochs -- number of iterations for the optimization loop
    print_cost -- if set to True, this will print the cost every 100 iterations
    lambd -- the regularization factor

    Returns:
    parameters -- a dictionary containing W1, W2, b1, and b2
    """


    costs = []  # to keep track of the cost
    m = X.shape[1]  # number of examples
    (n_x, n_h, n_y) = layers_dims  # unboxing the layer dimensions

    # Initialising hyperparameters for Adam
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8

    # Randomly partition the batches
    mini_batches = list()
    permutation = list(np.random.permutation(m))  # Random permutation
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation]
    number_complete_minibatches = math.floor(m / mini_batch_size)
    for k in range(0, number_complete_minibatches):
        mini_batch_X = shuffled_X[:, mini_batch_size * k : mini_batch_size * (k+1)]
        mini_batch_Y = shuffled_Y[:, mini_batch_size * k : mini_batch_size * (k+1)]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the case in which the last batch is not as big as the other batches
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:,
                       mini_batch_size * (number_complete_minibatches - 1): mini_batch_size * number_complete_minibatches]
        mini_batch_Y = shuffled_Y[:,
                       mini_batch_size * (number_complete_minibatches - 1): mini_batch_size * number_complete_minibatches]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # mini_batches contains now the batches we are going to work with

    # Initialise the parameters randomly
    parameters = {}
    dimensions = [n_x] + n_h + [n_y]
    for l in range(1, len(dimensions)):  # Using He Initialization
        parameters['W' + str(l)] = np.random.randn(dimensions[l], dimensions[l-1]) * np.sqrt(2 / dimensions[l-1])
        parameters['b' + str(l)] = np.zeros((dimensions[l], 1))

    # Start Optimization - Mini-Batch Gradient Descent
    activations = {}  # Dictionary to hold the activation vectors ( A_0, A_1, ... A_(L-1) )
    cache = {}  # Dictionary in which we cache the results of Z (we need them for back propagation algorithm)
    for i in range(0, num_epochs):
        # learning_rate = learning_rate / (1 + decay_rate * (i + 1))  # Decaying the learning rate
        for j in range(len(mini_batches)):
            mini_batch_X, mini_batch_Y = mini_batches[j]
            # Forward Propagation Step
            activations['A' + str(0)] = mini_batch_X
            for k in range(1, len(dimensions)):
                A_prev = activations['A' + str(k-1)]  # Take the previous activation vector (initially the input vector)
                W = parameters['W' + str(k)]  # Retrieve the weights from the parameters dictionary
                b = parameters['b' + str(k)]  # Retrieve the bias from the parameters dictionary
                Z = np.dot(W, A_prev) + b  # Compute the Z value for the current layer
                if k == len(dimensions) - 1:  # Compute the activation for the current layer
                    A = sigmoid(Z)
                else:
                    A = leaky_relu(Z)
                activations['A' + str(k)] = A  # Store the activation so we can use it for the following iteration
                cache['Z' + str(k)] = Z  # Cache the current Z vector to be used later in the back propagation phase

            # Cost computation + Regularization
            A_last = A  # Taking the last activation vector
            cost = - 1 / m * np.sum(np.dot(mini_batch_Y.T, logarithm(A_last)) + np.dot((1-mini_batch_Y).T, logarithm(1-A_last)))
            regularization = 0  # Initialising the regularization
            for l in range(1, len(dimensions)):
                W = parameters['W' + str(l)]
                regularization += np.sum(np.square(W))

            cost += (lambd / m * regularization)
            costs.append(cost)

            # Backward Propagation Step
            grads = dict()  # Declaring an empty gradient dictionary
            grads['dA' + str(len(dimensions) - 1)] = - np.divide(mini_batch_Y, A_last) + np.divide(1 - mini_batch_Y, 1 - A_last)
            for k in range(len(dimensions) - 1, 0, -1):
                dA = grads['dA' + str(k)]
                if k == len(dimensions) - 1:
                    dZ = dA * sigmoid_derivative(cache['Z' + str(k)])
                else:
                    dZ = dA * relu_derivative(cache['Z' + str(k)])
                A_prev = activations['A' + str(k-1)]
                W = parameters['W' + str(k)]
                dW = 1 / m * np.dot(dZ, A_prev.T) + lambd / m * W
                db = 1 / m * np.sum(dZ, axis=1, keepdims=True)
                grads["dW" + str(k)] = dW
                grads["db" + str(k)] = db
                dA_prev = np.dot(W.T, dZ)
                grads['dA' + str(k-1)] = dA_prev

            # Updating parameters
            #for l in range(len(dimensions)-1):
                #parameters["W" + str(l + 1)] -= learning_rate * grads["dW" + str(l + 1)]
                #parameters["b" + str(l + 1)] -= learning_rate * grads["db" + str(l + 1)]
                # Updating parameters - Adam Optimizer
            v, s = initialize_adam(parameters, dimensions)  # Initialising parameters v and s
            t = j+1  # Current iteration (+1 because the first iteration is 1, not 0)
            v_corrected = {}  # Initialising dictionary for corrected v values (bias correction)
            s_corrected = {}  # Initialising dictionary for corrected s values (bias correction)
            for l in range(len(dimensions) - 1):
                v['dW' + str(l + 1)] = beta1 * v['dW' + str(l + 1)] + (1 - beta1) * grads['dW' + str(l + 1)]
                v['db' + str(l + 1)] = beta1 * v['db' + str(l + 1)] + (1 - beta1) * grads['db' + str(l + 1)]
                v_corrected['dW' + str(l + 1)] = v['dW' + str(l + 1)] / (1 - np.power(beta1, t))
                v_corrected['db' + str(l + 1)] = v['db' + str(l + 1)] / (1 - np.power(beta1, t))
                s['dW' + str(l + 1)] = beta2 * s['dW' + str(l + 1)] + (1 - beta2) * np.power(grads['dW' + str(l + 1)], 2)
                s['db' + str(l + 1)] = beta2 * s['db' + str(l + 1)] + (1 - beta2) * np.power(grads['db' + str(l + 1)], 2)
                s_corrected['dW' + str(l + 1)] = s['dW' + str(l + 1)] / (1 - np.power(beta2, t))
                s_corrected['db' + str(l + 1)] = s['db' + str(l + 1)] / (1 - np.power(beta2, t))
                parameters["W" + str(l + 1)] -= learning_rate * v_corrected['dW' + str(l + 1)] / np.sqrt(s_corrected['dW' + str(l + 1)] + epsilon)
                parameters["b" + str(l + 1)] -= learning_rate * v_corrected['db' + str(l + 1)] / np.sqrt(s_corrected['db' + str(l + 1)] + epsilon)
                #parameters["W" + str(l + 1)] -= learning_rate * grads["dW" + str(l + 1)]
                #parameters["b" + str(l + 1)] -= learning_rate * grads["db" + str(l + 1)]

            # Print the cost after every iteration
            if print_cost:
                print("Cost after iteration {}: {}".format(j, np.squeeze(cost)))
        print("Finished epoch {}".format(i))


    # Plot the cost graph at the end

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('# iterations')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters
