import numpy as np
import csv
from model import model
from featureScaling import featureScaling
from predict import predict

'''LOADING THE TRAINING DATA SET + PREPROCESSING'''
X_train_original = []
Y_train_original = []
# Reading the data from the file
m_train = 8000  # Number of training data entries (max = 27456 with row 0 of labels)
n_x = 784  # Number of features (28 x 28 = 784)
index = 0
with open('./data/sign_mnist_train.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        if index > 0:
            X_train_original.append(row[1:])
            Y_train_original.append(row[0:1])
        index += 1
        if index > m_train:
            break

# Transforming the arrays to a numpy array
X_train = np.array(X_train_original).T  # n_x x m_train
Y_train = np.array(Y_train_original).T  # 1 x m_train
X_train = X_train.astype('float64')  # Changing the dtypes to float64
Y_train = Y_train.astype('float64')

# Transforming the Y_train array to a "true label" array - 1 for Yes, 0 for No
rows = []
for i in range(len(Y_train[0])):  # For each training example
    el = int(Y_train[0][i])  # Storing the current element in 'el'
    row = [0] * 25  # Creating an array full of zeros with 25 entries (for each letter)
    row[el] = 1  # Marking the current letter with 1
    rows.append(row)  # Appending the row to the array of rows
Y_train = np.array(rows).T  # Transforming the array of rows into a numpy array

'''SCALING FEATURES'''
[X_train, lmbda, mu] = featureScaling(X_train)

'''STRUCTURING THE ARTIFICIAL NEURAL NETWORK'''
n_x = 784 # Number of input units
n_h = [n_x * 2, n_x * 2, n_x * 2, n_x * 2]  # Array with the layers - 4 hidden layers - n_x * 2 units each
n_y = 25  # Number of output units (# of letters in the English Alphabet without Z)
layers_dims = (n_x, n_h, n_y)  # Grouping the dimensions in a tuple

'''TRAINING THE ARTIFICIAL NEURAL NETWORK MODEL'''
learning_rate = 0.005  # Initialising the Learning Rate
num_iterations = 30  # Setting the number of iterations
lambd_reg = 0.3  # Setting the regularization factor
parameters = model(X_train, Y_train, layers_dims, learning_rate, num_iterations, True, lambd_reg)

'''ACCURACY PREDICTION FOR THE TRAINING SET'''
train_accurracy = predict(X_train, Y_train, parameters, layers_dims)
print("Training accuracy: " + str(train_accurracy) + "%")

'''LOADING THE TEST DATA SET + PREPROCESSING'''
X_test_original = []
Y_test_original = []
# Reading the data from the file
m_test = 3000  # Number of test data entries (max = 7172 with row 0 of labels)
index = 0
with open('./data/sign_mnist_test.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        if index > 0:
            X_test_original.append(row[1:])
            Y_test_original.append(row[0:1])
        index += 1
        if index > m_train:
            break

# Transforming the arrays to a numpy array
X_test = np.array(X_test_original).T  # n_x x m_test
Y_test = np.array(Y_test_original).T  # 1 x m_test
X_test = X_test.astype('float64')  # Changing the dtypes to float64
Y_test = Y_test.astype('float64')

# Transforming the Y_test array to a "true label" array - 1 for Yes, 0 for No
rows = []
for i in range(len(Y_test[0])):  # For each test example
    el = int(Y_test[0][i])  # Storing the current element in 'el'
    row = [0] * 25  # Creating an array full of zeros with 25 entries (for each letter)
    row[el] = 1  # Marking the current letter with 1
    rows.append(row)  # Appending the row to the array of rows
Y_test = np.array(rows).T  # Transforming the array of rows into a numpy array

# Scale features using the previously computed lambda and miu from the training set
X_test = (X_test - lmbda) / mu

'''ACCURACY PREDICTION FOR THE TEST SET'''
test_accurracy = predict(X_test, Y_test, parameters, layers_dims)
print("Training accuracy: " + str(test_accurracy) + "%")
