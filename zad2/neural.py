from enum import Enum
import math
import numpy as np
from scipy.special import softmax as axis_softmax

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))

def d_sigmoid(y: np.ndarray) -> np.ndarray:
    return y * (1 - y)

def tanh(x: np.ndarray) -> np.ndarray:
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def d_tanh(y: np.ndarray) -> np.ndarray:
    return 1 - np.square(y)

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)

def d_relu(y: np.ndarray) -> np.ndarray:
    return np.where(y > 0, 1, 0)

def leaky_relu(x: np.ndarray, alpha = 0.01) -> np.ndarray:
    return np.where(x > 0, x, alpha * x)

def d_leaky_relu(y: np.ndarray, alpha = 0.01) -> np.ndarray:
    return np.where(y > 0, 1, alpha)

def softmax(x: np.ndarray) -> np.ndarray:
    return axis_softmax(x, axis = 0)

class Activation(Enum):
    SIGMOID = 1
    TANH = 2
    RELU = 3
    LEAKY_RELU = 4
    SOFTMAX = 5

class Layer:
    def __init__(self: 'Layer', input_size: int, layer_size: int, activation: Activation, learning_rate: float):
        """
        Initialise the network layer

        Parameters:
        input_size (int): The size of the input vector for this layer
        layer_size (int): The number of neurons this layer has
        learning_rate (float): A scaling factor for gradient descent during back propagation
        activation (Activation): The activation function for this layer
        """
        self.learning_rate = learning_rate

        if activation == Activation.SIGMOID:
            self.activation = sigmoid
            self.d_activation = d_sigmoid
        elif activation == Activation.TANH:
            self.activation = tanh
            self.d_activation = d_tanh
        elif activation == Activation.RELU:
            self.activation = relu
            self.d_activation = d_relu
        elif activation == Activation.LEAKY_RELU:
            self.activation = leaky_relu
            self.d_activation = d_leaky_relu
        elif activation == Activation.SOFTMAX:
            self.activation = softmax
            self.d_activation = None
        else:
            raise ValueError("Invalid activation function")

        self.input_size = input_size
        input_size_with_bias = input_size + 1

        weights_shape = (layer_size, input_size_with_bias)
        self.weights = np.random.normal(0, 1 / math.sqrt(input_size_with_bias), weights_shape)

    def forward_propagate(self: 'Layer', data: np.ndarray) -> np.ndarray:
        """
        Calculate the output of the layer for the given data

        Parameters:
        data (np.ndarray): The input data; should have shape (self.input_size, batch_size)
        """
        _, batch_size = data.shape
        self.batch_size = batch_size

        data_with_bias = np.vstack((np.ones((1, batch_size)), data))
        self.data_with_bias = data_with_bias

        self.net = self.weights @ data_with_bias
        self.result = self.__activate(self.net)
        return self.result

    def backward_propagate(self: 'Layer', d_loss: np.ndarray) -> np.ndarray:
        """
        Modify the weights based on the derivative of a loss function

        Parameters:
        d_loss (np.ndarray): The derivatives of the loss function by each of the outputs of
                             this layer; should have shape (self.layer_size, 1)
        """
        # the argument in the mathematical sense is self.net,
        # but the derivative is dependent only on self.result
        delta = d_loss * self.__d_activate(self.result)
        d_weights = delta @ self.data_with_bias.T / self.batch_size

        self.weights -= self.learning_rate * d_weights

        return (self.weights.T @ delta)[1:]

    def backward_propagate_softmax(self: 'Layer', result: np.ndarray, expected: np.ndarray) -> np.ndarray:
        """
        Modify the weights based on the categorical cross entropy function and softmax on the last layer

        Parameters:
        result (np.ndarray): The result of the layer for the inputs
        expected (np.ndarray): The expected result for the inputs
        """
        # print(result)
        # print(expected)
        delta = result - expected
        d_weights = delta @ self.data_with_bias.T / self.batch_size

        self.weights -= self.learning_rate * d_weights

        return (self.weights.T @ delta)[1:]

    def __activate(self: 'Layer', net: np.ndarray) -> np.ndarray:
        return self.activation(net)

    def __d_activate(self: 'Layer', result: np.ndarray) -> np.ndarray:
        return self.d_activation(result)

class NeuralNetwork:
    def __init__(self: 'NeuralNetwork', input_size: int, no_layers: int, no_neurons: np.ndarray, activations = np.ndarray, learning_rate = 0.01):
        self.layers = [
            Layer(input_size, no_neurons[0], activations[0], learning_rate) if i == 0
            else Layer(no_neurons[i - 1], no_neurons[i], activations[i], learning_rate)
            for i in range(no_layers)
        ]

        if activations[-1] == Activation.SOFTMAX:
            self.__backward_propagate_last = self.__backward_propagate_cce
        else:
            self.__backward_propagate_last = self.__backward_propagate_mse

    def fit(self: 'NeuralNetwork', data: np.ndarray, expected: np.ndarray):
        result = self.__forward_propagate(data)
        d_loss = self.__backward_propagate_last(result, expected)
        self.__backward_propagate(d_loss)

    def predict(self: 'NeuralNetwork', data: np.ndarray) -> np.ndarray:
        return self.__forward_propagate(data)

    def __forward_propagate(self: 'NeuralNetwork', data: np.ndarray) -> np.ndarray:
        result = data
        for layer in self.layers:
            result = layer.forward_propagate(result)

        return result

    def __backward_propagate(self: 'NeuralNetwork', d_loss: np.ndarray):
        for layer in reversed(self.layers[:-1]):
            d_loss = layer.backward_propagate(d_loss)

    def __backward_propagate_mse(self: 'NeuralNetwork', result: np.ndarray, expected: np.ndarray) -> np.ndarray:
        d_loss = result - expected
        return self.layers[-1].backward_propagate(d_loss)

    def __backward_propagate_cce(self: 'NeuralNetwork', result: np.ndarray, expected: np.ndarray) -> np.ndarray:
        return self.layers[-1].backward_propagate_softmax(result, expected)
