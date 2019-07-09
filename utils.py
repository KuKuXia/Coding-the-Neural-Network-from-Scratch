import struct
import numpy as np
import matplotlib.pyplot as plt


# Load the MNIST dataset
def load_data():
    with open('train-labels.idx1-ubyte', 'rb') as labels:
        magic, n = struct.unpack('>II', labels.read(8))
        train_labels = np.fromfile(labels, dtype=np.uint8)
    with open('train-images.idx3-ubyte', 'rb') as images:
        magic, num, nrows, ncols = struct.unpack('>IIII', images.read(16))
        train_images = np.fromfile(images, dtype=np.uint8).reshape(num, 784)
    with open('t10k-labels.idx1-ubyte', 'rb') as labels:
        magic, n = struct.unpack('>II', labels.read(8))
        test_labels = np.fromfile(labels, dtype=np.uint8)
    with open('t10k-images.idx3-ubyte', 'rb') as images:
        magic, num, nrows, ncols = struct.unpack('>IIII', images.read(16))
        test_images = np.fromfile(images, dtype=np.uint8).reshape(num, 784)
    return train_images, train_labels, test_images, test_labels


# Display the data
def visualize_data(img_array, label_array):
    fig, ax = plt.subplots(nrows=8, ncols=8, sharex=True, sharey=True)
    ax = ax.flatten()
    for i in range(64):
        img = img_array[label_array == 7][i].reshape(28, 28)
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')
    plt.show()


# The sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# Show the sigmoid function
def visualize_sigmoid():
    x = np.arange(-10, 10, 0.1)
    y = sigmoid(x)
    fig, ax = plt.subplots()
    ax.plot(x, y)
    plt.show()


# Plot the result
def plot_cost_and_accuracy(cost, acc):
    x_a = [i for i in range(acc.shape[0])]
    x_c = [i for i in range(len(cost))]
    print('final prediction accuracy is: ', acc[-1])
    plt.subplot(221)
    plt.plot(x_c, cost)
    plt.subplot(222)
    plt.plot(x_a, acc)
    plt.show()


# Plot the mis-classified images
def show_mis_classified_images(mis_img, correct_label, mis_label):
    fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True)
    ax = ax.flatten()
    for i in range(25):
        img = mis_img[i].reshape(28, 28)
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')
        ax[i].set_title('%d) t: %d p: %d' %
                        (i + 1, correct_label[i], mis_label[i]))
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()
