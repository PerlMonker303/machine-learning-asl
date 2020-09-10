from auxiliary import *

def checkGradient(parameters, grads, X, Y, dimensions, lambd, epsilon = 1e-7):
    """
    Computes the gradients explicitly and compares this to the gradients computed using the back propagation algorithm
    Arguments:
        parameters -- dictionary of parameters
        grads -- dictionary of gradients computed with the back propagation algorithm
        X -- input data, of shape (n_x, number of examples)
        Y -- vector of elements from 0 to 24 (corresponding to each letter of the English Alphabet)
        dimensions -- array containing the dimensions of each layer
        lambd -- The regularization factor
        epsilon -- constant used for computing the derivatives explicitly

    Returns:
        difference -- the difference between the gradients (from backprop and from explicit calculation)
    """

    parameters_vector = dictionary_to_vector(parameters)  # Taking the vector of parameters
    grads_vector = dictionary_to_vector(grads)  # Taking the vector of gradients
    n_x = X.shape[0]  # Number of parameters
    m = X.shape[1]  # Number of examples
    J_plus = np.zeros((n_x, 1))  # Initialise the positive cost vector
    J_minus = np.zeros((n_x, 1))  # Initialise the negative cost vector
    gradapprox = np.zeros((n_x, 1))  # Initialise the gradient approximation vector

    for i in range(int(len(parameters) / 2)):  # For each parameter
        # For thetaplus
        thetaplus = np.copy(parameters_vector)
        thetaplus[i][0] = thetaplus[i][0] + epsilon
        # For thetaminus
        thetaminus = np.copy(parameters_vector)
        thetaminus[i][0] = thetaminus[i][0] - epsilon

        # Feedforward for computing the cost function
        parameters_forward_plus = vector_to_dictionary(thetaplus)
        parameters_forward_minus = vector_to_dictionary(thetaminus)
        activations_plus = {}
        activations_minus = {}
        activations_plus['A' + str(0)] = X
        activations_minus['A' + str(0)] = X
        for j in range(1, len(dimensions)):
            A_prev_plus = activations_plus['A' + str(j - 1)]  # Take the previous activation vector (initially the input vector)
            A_prev_minus = activations_minus['A' + str(j - 1)]  # Take the previous activation vector (initially the input vector)
            W_plus = parameters_forward_plus['W' + str(j)]  # Retrieve the weights from the parameters dictionary
            W_minus = parameters_forward_minus['W' + str(j)]  # Retrieve the weights from the parameters dictionary
            b_plus = parameters_forward_plus['b' + str(j)]  # Retrieve the bias from the parameters dictionary
            b_minus = parameters_forward_minus['b' + str(j)]  # Retrieve the bias from the parameters dictionary
            Z_plus = np.dot(W_plus, A_prev_plus) + b_plus  # Compute the Z value for the current layer
            Z_minus = np.dot(W_minus, A_prev_minus) + b_minus  # Compute the Z value for the current layer
            if j == len(dimensions) - 1:  # Compute the activation for the current layer
                A_plus = sigmoid(Z_plus)
                A_minus = sigmoid(Z_minus)
            else:
                A_plus = leaky_relu(Z_plus)
                A_minus = leaky_relu(Z_minus)
            activations_plus['A' + str(j)] = A_plus  # Store the activation so we can use it for the following iteration
            activations_minus['A' + str(j)] = A_minus  # Store the activation so we can use it for the following iteration
        J_plus[i] = - 1 / m * np.sum(np.dot(Y.T, logarithm(A_plus)) + np.dot((1 - Y).T, logarithm(1 - A_plus)))
        J_minus[i] = - 1 / m * np.sum(np.dot(Y.T, logarithm(A_minus)) + np.dot((1 - Y).T, logarithm(1 - A_minus)))
        regularization = 0  # Initialising the regularization
        for l in range(1, len(dimensions)):
            W = parameters['W' + str(l)]
            regularization += np.sum(np.square(W))

        J_plus[i] += (lambd / m * regularization)
        J_minus[i] += (lambd / m * regularization)

        gradapprox[i] = (J_plus[i] - J_minus[i]) / (2 * epsilon)

    difference = 0
    numerator = np.linalg.norm(grads_vector - gradapprox)
    denominator = np.linalg.norm(grads_vector) + np.linalg.norm(gradapprox)
    difference = numerator / denominator

    print(difference)
    if difference > epsilon:
        print("SOMETHING IS FISHY IN THE BACKWARD PROPAGATION IMPLEMENTATION")
    else:
        print("PERFECT BACKWARD PROPAGATION IMPLEMENTATION")
    return difference
