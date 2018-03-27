# -*- coding: utf-8 -*-
"""An example of use of FF class."""
import numpy as np

from feed_forward import FF


def get_dataset1(row_per_class=100):
    """Generate a basic dataset with 2 classes.
    |    o o
    | x   o
    | x x
    |________

    :param row_per_class: number of row per class
    :type row_per_class: int
    :return: The features array and the targets array
    :rtype: (numpy.ndarray, numpy.ndarray)
    """
    class1 = np.random.randn(row_per_class, 2) + np.array([-2, -2])
    class2 = np.random.randn(row_per_class, 2) + np.array([2, 2])

    features = np.vstack([class1, class2])

    targets = np.concatenate((np.zeros(row_per_class),
                              np.zeros(row_per_class) + 1))

    return features, targets


def get_dataset2(row_per_class=100):
    """Generate a dataset with 2 classes.
    |   o  o
    | o o  o o
    |
    | x x  o o
    |   x  o
    |___________

    :param row_per_class: number of row per class
    :type row_per_class: int
    :return: The features array and the targets array
    :rtype: (numpy.ndarray, numpy.ndarray)
    """
    class1_1 = np.random.randn(row_per_class, 2) + np.array([-2, -2])

    class2_1 = np.random.randn(row_per_class, 2) + np.array([2, 2])
    class2_2 = np.random.randn(row_per_class, 2) + np.array([-2, 2])
    class2_3 = np.random.randn(row_per_class, 2) + np.array([2, -2])

    features = np.vstack([class1_1, class2_1, class2_2, class2_3])
    targets = np.concatenate((np.zeros(row_per_class * 2),
                              np.zeros(row_per_class * 2) + 1))

    return features, targets


def get_dataset3(row_per_class=100):
    """Generate a dataset with 2 classes.
    |   o  x
    | o o  x x
    |
    | x x  o o
    |   x  o
    |___________

    :param row_per_class: number of row per class
    :type row_per_class: int
    :return: The features array and the targets array
    :rtype: (numpy.ndarray, numpy.ndarray)
    """
    class1_1 = np.random.randn(row_per_class, 2) + np.array([-2, -2])
    class1_2 = np.random.randn(row_per_class, 2) + np.array([2, 2])

    class2_1 = np.random.randn(row_per_class, 2) + np.array([-2, 2])
    class2_2 = np.random.randn(row_per_class, 2) + np.array([2, -2])

    features = np.vstack([class1_1, class1_2, class2_1, class2_2])
    targets = np.concatenate((np.zeros(row_per_class * 2),
                              np.zeros(row_per_class * 2) + 1))

    return features, targets


def example1():
    """FF with 1 hidden layer of 2 neurons using dataset1."""
    features, targets = get_dataset1()
    ff = FF(nb_features=2, hidden_layer_size=2)
    ff.load_dataset(features, targets)
    ff.train(epochs=100, learning_rate=0.1)
    ff.draw()


def example2():
    """FF with 1 hidden layer of 5 neurons using dataset2."""
    features, targets = get_dataset2()
    ff = FF(nb_features=2, hidden_layer_size=5)
    ff.load_dataset(features, targets)
    ff.train(epochs=100, learning_rate=0.1)
    ff.draw()


def example3():
    """FF with 1 hidden layer of 3 neurons using dataset3."""
    features, targets = get_dataset3()
    ff = FF(nb_features=2, hidden_layer_size=4)
    ff.load_dataset(features, targets)
    ff.train(epochs=100, learning_rate=0.1)
    ff.draw()


if __name__ == '__main__':
    """Launch example2."""
    example2()
