# -*- coding: utf-8 -*-
"""A very basic feed forward neuronal network in python."""
import numpy as np
from viznet import connecta2a, node_sequence, NodeBrush, EdgeBrush, DynamicShow


VIZNET_LAYER_TPL = """{layer_type} layer weights:
{weights}

{layer_type} layer biases:
{biases}"""


class FF(object):
    """A Feed forward neuronal network using grandient descent."""

    def __init__(self, nb_features, hidden_layer_size):
        """Initialize network weights and biases.

        :param nb_features: The number of features in the dataset
        :param hidden_layer_size: The number of neurons in the hidden layer
        :type nb_features: int
        :type hidden_layer_size: int
        """
        self.w1 = np.random.randn(nb_features, hidden_layer_size)
        self.b1 = np.zeros(hidden_layer_size)
        self.w2 = np.random.randn(hidden_layer_size)
        self.b2 = np.zeros(1)
        self.is_trained = False

    @staticmethod
    def __activation(z):
        """Neuronal activation method.

        :param: z: The result of the pre-activation
        :type z: numpy.ndarray
        :return: The result of the activation
        :rtype: numpy.ndarray
        """
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def __pre_activation(dataset, weights, biases):
        """Neuronal pre-activation method.

        :param dataset: The loaded dataset
        :param weights: All neurons weights
        :param biases: All neurons biases
        :type dataset: numpy.ndarray
        :type weights: numpy.ndarray
        :type biases: numpy.ndarray
        :return: The result of the pre-activation
        :rtype: numpy.ndarray
        """
        return np.dot(dataset, weights) + biases

    @classmethod
    def _derivative_activation(cls, z):
        """The derivative of the activation function.

        This function is used by the gradient descent algorithm.

        :param z: The result of the activation method of the previous layer.
        :type z: numpy.ndarray
        :return: The result of the derivation of the activation method.
        :rtype: numpy.ndarray
        """
        return cls.__activation(z) * (1 - cls.__activation(z))

    def _compute_performance(self):
        """Compute the performance of the current network.

        Compute a class_prediction array and a cost array and store
        them as instance variables.

        We manually compute an euclidian distance.
        """
        z1 = self.__pre_activation(self.dataset, self.w1, self.b1)
        a1 = self.__activation(z1)
        z2 = self.__pre_activation(a1, self.w2, self.b2)

        self.predictions = self.__activation(z2)
        self.class_predictions = np.round(self.predictions)
        self.accuracy = np.mean(self.class_predictions == self.targets)

    def draw(self):
        """Draw the feed forward network.

        Display the network, with neuron weights and biases.
        Display also the performance of the network.
        """
        num_node_list = [self.w1.shape[0], self.w1.shape[1], 1]
        with DynamicShow((13, 6)) as d:
            kind_list = ["nn.input", "nn.hidden", "nn.output"]
            y_list = 1.5 * np.arange(3)
            seq_list = []
            for n, kind, y in zip(num_node_list, kind_list, y_list):
                b = NodeBrush(kind, d.ax)
                seq_list.append(node_sequence(b, n, center=(0, y)))

            eb = EdgeBrush('-->', d.ax)
            for st, et in zip(seq_list[:-1], seq_list[1:]):
                connecta2a(st, et, eb)

            d.ax.set_title("Feed forward network with 1 hidden layer "
                           "of %s neurons" % self.w1.shape[1],
                           fontdict={"fontsize": 17})
            d.ax.text(-4, 3, "Accuracy: %d %%" % (self.accuracy * 100),
                      fontweight="bold",
                      bbox={"facecolor": "lightgrey", "alpha": 0.5, "pad": 10})
            d.ax.text(-4, 2.6, "Epochs: %s" % self.epochs)
            d.ax.text(-4, 2.4, "Learning rate: %s" % self.learning_rate)
            d.ax.text(-4, 2.2, "Nb rows: %s" % self.dataset.shape[0])
            w1_text = VIZNET_LAYER_TPL.format(
                layer_type="Hidden",
                weights=np.array_str(self.w1, precision=2),
                biases=np.array_str(self.b1, precision=2))
            d.ax.text(2.7, 0, w1_text,
                      bbox={"facecolor": "lightgrey", "alpha": 0.1, "pad": 10})
            w2_text = VIZNET_LAYER_TPL.format(
                layer_type="Output",
                weights=np.array_str(self.w2, precision=2),
                biases=np.array_str(self.b2, precision=2))
            d.ax.text(2.7, 2.5, w2_text,
                      bbox={"facecolor": "lightgrey", "alpha": 0.1, "pad": 10})

    def load_dataset(self, dataset, targets):
        """Load the dataset to the neuronal network.

        We compute the performance of the untrained network.

        :param dataset: The dataset
        :param targets: The targets (the expected results)
        :type dataset: numpy.ndarray
        :type targets: numpy.ndarray
        """
        assert isinstance(targets, np.ndarray), (
            "The target argument should be a numpy ndarray")
        expected_dataset_shape = (targets.shape[0], self.w1.shape[0])
        assert isinstance(dataset, np.ndarray), (
            "The dataset should be a numpy %n x %n numpy ndarray"
            % expected_dataset_shape)
        assert dataset.shape == expected_dataset_shape, (
            "The dataset should be a %n x %n numpy ndarray"
            % expected_dataset_shape)

        self.dataset = dataset
        self.targets = targets

    def train(self, epochs=100, learning_rate=0.1):
        """Train the neuronal network.

        We use the gradient descent method. The error function is computed
        using an euclidian distance.

        We compute the performance of the trained network.

        :param epochs: The number of epochs (default=100)
        :param learning_rate: The learning rate (default=0.1)
        :type epochs: int
        :type learning_rate: float
        """
        assert hasattr(self, "targets") and hasattr(self, "dataset"), (
            "Please load a dataset and targets first by calling "
            "load_dataset method")

        self.epochs = epochs
        self.learning_rate = learning_rate

        for epoch in range(epochs):
            # Init gradients
            w1_gradients = np.zeros(self.w1.shape)
            b1_gradients = np.zeros(self.b1.shape)
            w2_gradients = np.zeros(self.w2.shape)
            b2_gradients = np.zeros(self.b2.shape)
            # Go through each row
            for dataset, target in zip(self.dataset, self.targets):
                # Compute prediction
                z1 = self.__pre_activation(dataset, self.w1, self.b1)
                a1 = self.__activation(z1)
                z2 = self.__pre_activation(a1, self.w2, self.b2)
                predictions = self.__activation(z2)
                # Compute the error term
                error_term = (predictions - target)
                # Compute the error term for the output layer
                error_term_output = (error_term *
                                     self._derivative_activation(z2))
                # Compute the error_term for the hidden layer
                error_term_hidden = (error_term_output * self.w2 *
                                     self._derivative_activation(z1))
                # Update gradients
                w1_gradients += error_term_hidden * dataset[:, None]
                b1_gradients += error_term_hidden
                w2_gradients += error_term_output * a1
                b2_gradients += error_term_output
            # Update variables
            self.w1 -= learning_rate * w1_gradients
            self.b1 -= learning_rate * b1_gradients
            self.w2 -= learning_rate * w2_gradients
            self.b2 -= learning_rate * b2_gradients

        self.is_trained = True

        # Compute DNN performance
        self._compute_performance()
