from auxiliary import *

def predict(X, Y, parameters, layers_dims):
    """
    Predicts the accuracy of the Neural Network provided an input set, an output set and a set of optimized parameters

    Arguments:
    X -- input data, of shape (n_x, number of examples)
    Y -- vector of elements from 0 to 24 (corresponding to each letter of the English Alphabet)
    parameters -- a dictionary containing W1, W2, b1, and b2
    layers_dims -- dimensions of the layers (n_x, n_h, n_y)

    Returns:
    accuracy -- the accuracy of the algorithm (correct_guesses / size_input)
    """
    accuracy = 0  # Initialising the accuracy
    m = X.shape[1]  # Number of input data
    (n_x, n_h, n_y) = layers_dims  # unboxing the layer dimensions
    dimensions = [n_x] + n_h + [n_y]

    activations = {}  # Dictionary to hold the activation vectors ( A_0, A_1, ... A_(L-1) )
    activations['A' + str(0)] = X
    for j in range(1, len(dimensions)):
        A_prev = activations['A' + str(j - 1)]  # Take the previous activation vector (initially the input vector)
        W = parameters['W' + str(j)]  # Retrieve the weights from the parameters dictionary
        b = parameters['b' + str(j)]  # Retrieve the bias from the parameters dictionary
        Z = np.dot(W, A_prev) + b  # Compute the Z value for the current layer
        if j == len(dimensions) - 1:  # Compute the activation for the current layer
            A = sigmoid(Z)
        else:
            A = relu(Z)
        activations['A' + str(j)] = A  # Store the activation so we can use it for the following iteration

    A_last = A.T
    Y = Y.T
    count = 0
    for i in range(m):
        pos = np.where(Y[i] == 1)
        pos1 = int(pos[0][0])
        mx = np.max(A_last[i])
        pos = np.where(A_last[i] == mx)
        pos2 = int(pos[0][0])
        if pos2 == 16:  # PROBLEM: IT ALWAYS PREDICTS 16 INSTEAD OF THE RIGHT VALUE
            count += 1
        if pos1 == pos2:
            accuracy += 1
    print(count)
    print(m)
    return accuracy / m
