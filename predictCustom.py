from auxiliary import *
import matplotlib.pyplot as plt
from PIL import Image

def predictCustom(X_custom, m_custom, parameters, dimensions, X_custom_original):
    """
    Predicts the output for different custom images

    Arguments:
    X_custom -- input data, of shape (n_x, number of input examples)
    m_custom -- number of elements to predict
    parameters -- a dictionary containing weights (W) and biases (b)
    dimensions -- array of dimensions of the layers (n_x, n_h, ..., n_h, n_y)
    X_custom_original -- input data (without normalization) - used for showing the images

    Returns:
    """
    for i in range(m_custom):
        activations = {}
        # Forward Propagation Step
        activations['A' + str(0)] = X_custom
        for j in range(1, len(dimensions)):
            A_prev = activations['A' + str(j - 1)]  # Take the previous activation vector (initially the input vector)
            W = parameters['W' + str(j)]  # Retrieve the weights from the parameters dictionary
            b = parameters['b' + str(j)]  # Retrieve the bias from the parameters dictionary
            Z = np.dot(W, A_prev) + b  # Compute the Z value for the current layer
            if j == len(dimensions) - 1:  # Compute the activation for the current layer
                A = sigmoid(Z)
            else:
                A = leaky_relu(Z)
            activations['A' + str(j)] = A  # Store the activation so we can use it for the following iteration

        A_last = A.T
        mx = np.max(A_last[i])
        pos = np.where(A_last[i] == mx)
        pos = int(pos[0][0])
        arr = X_custom_original[:, i].reshape(28, 28).transpose()
        img = Image.fromarray(arr)
        plt.imshow(img)
        plt.show()
        print('CUSTOM PREDICTION for image with id ' + str(i+1) + ': ' + str(chr(65 + pos)))
