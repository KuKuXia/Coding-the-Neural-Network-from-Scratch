"""
Neural Network from Scratch using Python
"""

# Import the library
import os
import struct

import matplotlib.pyplot as plt
import numpy as np

from utils import *


# One hot encoding for category labels
def one_hot_encoding(y, num_labels=10):
    one_hot = np.zeros((num_labels, y.shape[0]))
    for i, val in enumerate(y):
        one_hot[val, i] = 1.0
    return one_hot


# The gradient of the sigmoid function
def sigmoid_gradient(z):
    s = sigmoid(z)
    return s*(1-s)


# Calculate the cost function
def calculate_cost(y_encoded, output):
    t1 = -y_encoded * np.log(output)
    t2 = (1 - y_encoded) * np.log(1 - output)
    return np.sum(t1 - t2)


# Add the bias unit, where is just row or column
def add_bias_unit(X, where):
    if where == 'column':
        X_new = np.ones((X.shape[0], X.shape[1] + 1))
        X_new[:, 1:] = X
    elif where == 'row':
        X_new = np.ones((X.shape[0] + 1, X.shape[1]))
        X_new[1:, :] = X
    return X_new


# Initializing the weights
def init_weights(n_features, n_hidden, n_output):
    w1 = np.random.uniform(-1.0, 1.0, size=n_hidden*(n_features+1))
    w1 = w1.reshape(n_hidden, n_features+1)
    w2 = np.random.uniform(-1.0, 1.0, size=n_hidden*(n_hidden+1))
    w2 = w2.reshape(n_hidden, n_hidden+1)
    w3 = np.random.uniform(-1.0, 1.0, size=n_output*(n_hidden+1))
    w3 = w3.reshape(n_output, n_hidden+1)
    return w1, w2, w3


# Define the feed forward function
def feed_forward(x, w1, w2, w3):
    # add bias unit to the input, column within the  row is just a byte of data, so we need to add a column vector of ones
    a1 = add_bias_unit(x, where='column')
    z2 = w1.dot(a1.T)
    a2 = sigmoid(z2)

    # since we transposed we have to add bias units as a row
    a2 = add_bias_unit(a2, where='row')
    z3 = w2.dot(a2)
    a3 = sigmoid(z3)
    a3 = add_bias_unit(a3, where='row')
    z4 = w3.dot(a3)
    a4 = sigmoid(z4)

    return a1, z2, a2, z3, a3, z4, a4


# Predict using the weights
def predict(x, w1, w2, w3):
    a1, z2, a2, z3, a3, z4, a4 = feed_forward(x, w1, w2, w3)
    y_pred = np.argmax(a4, axis=0)
    return y_pred


# Calculate the gradient
def calculate_gradient(a1, a2, a3, a4, z2, z3, z4, y_enc, w1, w2, w3):
    delta4 = a4 - y_enc
    z3 = add_bias_unit(z3, where='row')
    delta3 = w3.T.dot(delta4)*sigmoid_gradient(z3)
    delta3 = delta3[1:, :]
    z2 = add_bias_unit(z2, where='row')
    delta2 = w2.T.dot(delta3)*sigmoid_gradient(z2)
    delta2 = delta2[1:, :]

    grad1 = delta2.dot(a1)
    grad2 = delta3.dot(a2.T)
    grad3 = delta4.dot(a3.T)

    return grad1, grad2, grad3


# Define the model
def runModel(X, y, X_t, y_t):
    # Copy the data
    X_copy, y_copy = X.copy(), y.copy()

    # One hot encoding
    y_enc = one_hot_encoding(y)
    epochs = 100
    batch = 50

    # Initialize the weights
    w1, w2, w3 = init_weights(784, 75, 10)

    alpha = 0.001
    eta = 0.001
    dec = 0.00001
    delta_w1_prev = np.zeros(w1.shape)
    delta_w2_prev = np.zeros(w2.shape)
    delta_w3_prev = np.zeros(w3.shape)
    total_cost = []
    pred_acc = np.zeros(epochs)

    for i in range(epochs):
        # Shuffle the dataset
        shuffle = np.random.permutation(y_copy.shape[0])
        X_copy, y_enc = X_copy[shuffle], y_enc[:, shuffle]
        eta /= (1 + dec*i)

        mini = np.array_split(range(y_copy.shape[0]), batch)

        for step in mini:
            # Feed the forward the model
            a1, z2, a2, z3, a3, z4, a4 = feed_forward(
                X_copy[step], w1, w2, w3)
            cost = calculate_cost(y_enc[:, step], a4)

            total_cost.append(cost)

            # Back propagate
            grad1, grad2, grad3 = calculate_gradient(a1, a2, a3, a4, z2, z3, z4, y_enc[:, step],
                                                     w1, w2, w3)

            # Update the weights
            delta_w1, delta_w2, delta_w3 = eta * grad1, eta * grad2, eta * grad3

            w1 -= delta_w1 + alpha * delta_w1_prev
            w2 -= delta_w2 + alpha * delta_w2_prev
            w3 -= delta_w3 + alpha * delta_w3_prev

            delta_w1_prev, delta_w2_prev, delta_w3_prev = delta_w1, delta_w2, delta_w3
        y_pred = predict(X_t, w1, w2, w3)
        pred_acc[i] = 100 * np.sum(y_t == y_pred, axis=0) / X_t.shape[0]
        print('epoch #', i)
    return total_cost, pred_acc, y_pred


# Load the data
train_x, train_y, test_x, test_y = load_data()

# Show the data
# visualize_data(train_x, train_y)

# Run the model
cost, acc, y_pred = runModel(train_x, train_y, test_x, test_y)

# Plot the result
plot_cost_and_accuracy(cost, acc)

# Find the mis-classified images
mis_img = test_x[test_y != y_pred][:25]
correct_label = test_y[test_y != y_pred][:25]
mis_label = y_pred[test_y != y_pred][:25]

show_mis_classified_images(mis_img, correct_label, mis_label)
