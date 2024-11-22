import math
import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))

def d_sigmoid(y: np.ndarray) -> np.ndarray:
    return y * (1 - y)

def tanh(x: np.ndarray) -> np.ndarray:
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def d_tanh(y: np.ndarray) -> np.ndarray:
    return 1 - y^2

def relu(x: np.ndarray) -> np.ndarray:
    return np.max(0, x)

def d_relu(y: np.ndarray) -> np.ndarray:
    return np.where(y > 0, 1, 0)

def leaky_relu(x: np.ndarray, alpha = 0.01) -> np.ndarray:
    return np.where(x > 0, x, alpha * x)

def d_leaky_relu(y: np.ndarray, alpha = 0.01) -> np.ndarray:
    return np.where(y > 0, 1, alpha)

def softmax(x: np.ndarray) -> np.ndarray:
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)

def d_mse(result: np.ndarray, expected: np.ndarray) -> np.ndarray:
    pass

def d_cce(result: np.ndarray, expected: np.ndarray) -> np.ndarray:
    pass

class Layer:
    def __init__(self: 'Layer', input_size: int, layer_size: int, learning_rate: float):
        """
        Initialise the network layer

        Parameters:
        input_size (int): The size of the input vector for this layer
        layer_size (int): The number of neurons this layer has
        learning_rate (float): A scaling factor for gradient descent during back propagation
        """
        self.learning_rate = learning_rate

        self.input_size = input_size
        input_size_with_bias = input_size + 1

        weights_shape = (layer_size, input_size_with_bias)
        self.weights = np.random.normal(0, 1 / math.sqrt(input_size_with_bias), weights_shape)

    def forward_propagate(self: 'Layer', data: np.ndarray) -> np.ndarray:
        """
        Calculate the output of the layer for the given data

        Parameters:
        data (np.ndarray): The input data; should have shape (self.input_size, 1)
        """
        data_with_bias = np.vstack(([1], data))
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
        delta = d_loss * self.__d_activate(self.result)
        d_weights = delta @ self.data_with_bias.T

        self.weights -= self.learning_rate * d_weights

        return (self.weights.T @ delta)[1:]

    def __activate(self: 'Layer', net: np.ndarray) -> np.ndarray:
        return sigmoid(net)

    def __d_activate(self: 'Layer', net: np.ndarray) -> np.ndarray:
        return d_sigmoid(net)

class NeuralNetwork:
    def __init__(self: 'NeuralNetwork', input_size: int, layers: int, neurons: np.ndarray, learning_rate = 1/100):
        self.layers = [
            Layer(input_size, neurons[0], learning_rate) if i == 0
            else Layer(neurons[i - 1], neurons[i], learning_rate)
            for i in range(layers)
        ]

    def fit(self: 'NeuralNetwork', data: np.ndarray, expected: np.ndarray):
        result = self.__forward_propagate(data)
        self.__backward_propagate(result, expected)

    def predict(self: 'NeuralNetwork', data: np.ndarray) -> np.ndarray:
        return self.__forward_propagate(data)

    def __forward_propagate(self: 'NeuralNetwork', data: np.ndarray) -> np.ndarray:
        result = data
        for layer in self.layers:
            result = layer.forward_propagate(result)

        return result

    def __backward_propagate(self: 'NeuralNetwork', result: np.ndarray, expected: np.ndarray):
        d_loss = result - expected
        for layer in reversed(self.layers):
            d_loss = layer.backward_propagate(d_loss)

# nn = NeuralNetwork(1, 3, [3, 3, 1])

# fizzbuzz_data = [
    # (i, 3) if i % 3 == 0 and i % 5 == 0 else
    # (i, 2) if i % 5 == 0 else
    # (i, 1) if i % 3 == 0 else
    # (i, 0)
    # for i in range(100000)
# ]
# fizzbuzz_test = [
    # (i, 3) if i % 3 == 0 and i % 5 == 0 else
    # (i, 2) if i % 5 == 0 else
    # (i, 1) if i % 3 == 0 else
    # (i, 0)
    # for i in range(150000, 160000)
# ]
# for i in range(10):
    # for data in fizzbuzz_data:
        # (d, e) = data
        # if d % 10000 == 0:
            # print(str(d) + "...")
        # nn.fit(d / 4, e)

# errors = 0

# for data in fizzbuzz_test:
    # (d, e) = data
    # r = nn.predict(d / 4)
    # print(d, "- got", r, ", expected", e)
    # if np.linalg.norm(r - e) > 0.1:
        # errors += 1

# print("errors:", errors / 10000)

# layer = Layer(3, 3, 1)
# print("weights:", layer.weights)

# data = np.array([[1/3], [2/3], [1]])
# expected = np.array([[4/6], [5/6], [1]])

# for _ in range(1, 1000):
    # result = layer.forward_propagate(data)
    # print("result: ", result)
    # d_loss = result - expected
    # print("d_loss: ", d_loss)
    # layer.backward_propagate(d_loss)
    # print("new weights:", layer.weights)

# result = layer.forward_propagate(data)
# print("final: ", result)
