import numpy as np
import csv
from model import model
from modelMiniBatch import modelMiniBatch
from featureScaling import featureScaling
from predict import predict
import matplotlib.pyplot as plt
import matplotlib.image as pimg
from PIL import Image
from predictCustom import predictCustom

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

'''VISUALISING SOME RANDOM IMAGES'''
np.random.seed()  # initialise a random seed
pictures_num = 0  # number of pictures we want to see
for i in range(pictures_num):
    rnd = np.random.randint(0, m_train)
    pic_vector = X_train[:, rnd].reshape(28, 28)
    plt.imshow(pic_vector)
    plt.show()

'''SCALING FEATURES'''
[X_train, lmbda, mu] = featureScaling(X_train)

'''STRUCTURING THE ARTIFICIAL NEURAL NETWORK'''
n_x = 784 # Number of input units
n_h = [n_x * 4, n_x * 4]  # Array with the layers - '2' hidden layers - n_x * 4 units each
n_y = 25  # Number of output units (# of letters in the English Alphabet without Z)
layers_dims = (n_x, n_h, n_y)  # Grouping the dimensions in a tuple

'''TRAINING THE ARTIFICIAL NEURAL NETWORK MODEL'''
learning_rate = 0.00003  # Initialising the Learning Rate
lambd_reg = 0.3  # Setting the regularization factor
mini_batch_size = 32  # Setting the size of a batch
num_epochs = 5  # Setting the number of epochs (= m => Batch G.D.; = 1 => Stochastic G.D.)
decay_rate = 1  # 0 if you don't want to use a decaying leraning rate
parameters = modelMiniBatch(X_train, Y_train, layers_dims, mini_batch_size, learning_rate, decay_rate, num_epochs, True, lambd_reg)

'''ACCURACY PREDICTION FOR THE TRAINING SET'''
train_accurracy = predict(X_train, Y_train, parameters, layers_dims)
print("Training accuracy: " + str(train_accurracy) + "%")

'''LOADING THE TEST DATA SET + PREPROCESSING'''
X_test_original = []
Y_test_original = []
# Reading the data from the file
m_test = 5000  # Number of test data entries (max = 7172 with row 0 of labels)
index = 0
with open('./data/sign_mnist_test.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        if index > 0:
            X_test_original.append(row[1:])
            Y_test_original.append(row[0:1])
        index += 1
        if index > m_test:
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

'''CUSTOM IMAGE PREDICTIONS'''

# Read the images
X_custom_original = []
m_custom = 6  # Number of custom images to predict
for i in range(m_custom):
    image = pimg.imread('./data/custom/cust_' + str(i+1) + '.jpg')
    arr = np.asarray(image)
    resized_img = Image.fromarray(arr).resize(size=(28, 28))
    arr = np.asarray(resized_img)
    X_custom_original.append(arr)

# Transforming the arrays to a numpy array
X_custom = np.array(X_custom_original).T  # n_x x m_custom
X_custom = X_custom.astype('float64')  # Changing the dtypes to float64

X_custom_colors = []
# Transform images from RGB to simple average
# Scale features using the previously computed lambda and miu from the training set
for i in range(m_custom):
    avg = (X_custom[0, :, :, i] + X_custom[1, :, :, i] + X_custom[2, :, :, i]) / 3
    avg = avg.reshape(784, 1)
    #avg = (avg - lmbda) / mu
    X_custom_colors.append(avg)

X_custom = np.array(X_custom_colors)
X_custom = X_custom.transpose(2, 0, 1).reshape(-1, X_custom.shape[1]).transpose()
X_custom_original = X_custom
X_custom = (X_custom - lmbda) / mu

# Custom prediction
dimensions = [n_x] + n_h + [n_y]  # Box together the dimensions
predictCustom(X_custom, m_custom, parameters, dimensions, X_custom_original)
