import numpy as np

def featureScaling(X):
    lmbda = np.mean(X, axis=1, keepdims=True)  # Compute the mean average for each input
    mu = np.std(X)  # Compute the Standard Deviation
    X_scaled = (X - lmbda) / mu  # Scale features
    return [X_scaled, lmbda, mu]


