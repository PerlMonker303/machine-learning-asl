import numpy as np

def featureScaling(X):
    """
    Scales the features X

    Arguments:
        X -- the input data to be scaled
    Returns:
        [X_scaled -- the scaled data,
        lmbda -- the mean average for each input,
        mu -- the standard deviation for each input]
    """
    lmbda = np.mean(X, axis=1, keepdims=True)  # Compute the mean average for each input
    mu = np.std(X)  # Compute the Standard Deviation
    X_scaled = (X - lmbda) / mu  # Scale features
    return [X_scaled, lmbda, mu]


