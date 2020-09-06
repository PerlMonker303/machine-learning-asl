import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import csv
from model import model

'''LOADING THE TRAINING DATA SET + PREPROCESSING'''
X_train_original = []
Y_train_original = []
# Reading the data from the file
m_train = 2000 # Number of training data entries (max = 27456 with row 0 of labels)
n_x = 784 # Number of features (28 x 28 = 784)
index = 0
with open('./data/sign_mnist_train.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        if index > 0:
            X_train_original.append(row[1:])
            Y_train_original.append(row[0:1])
        index +=1
        if index > m_train:
            break

# Transforming the array to a numpy array (NOTE: values are stored as strings)
X_train = np.array(X_train_original).T # n_x x m_train
Y_train = np.array(Y_train_original).T # 1 x m_train

'''STRUCTURING THE ARTIFICIAL NEURAL NETWORK'''
n_x = m_train # Number of input units
n_h = [n_x, n_x, n_x] # Array with the layers - 3 hidden layers - n_x units each
n_y = 26 # Number of output units (# of letters in the English Alphabet)
layers_dims = (n_x, n_h, n_y) # Grouping the dimensions in a tuple

'''CALLING THE ARTIFICIAL NEURAL NETWORK MODEL'''
learning_rate = 0.0075 # Initialising the Learning Rate
num_iterations = 500 # Setting the number of iterations
parameters = model(X_train, Y_train, layers_dims, learning_rate, num_iterations, True)

