from neural import Activation, Layer, NeuralNetwork
import numpy as np
import mnist_loader

# (training_data, validation_data, test_data) = mnist_loader.load_data_wrapper()

# nn = NeuralNetwork(784, 3, [100, 100, 10], [Activation.LEAKY_RELU, Activation.LEAKY_RELU, Activation.SOFTMAX], learning_rate = 0.0005)

# training_data = list(training_data)

# for i in range(10):
    # print("epoch", i + 1)
    # for data in training_data:
        # (img, expected) = data
        # print(img)
        # nn.fit(img, expected)

# errors = 0

# for data in test_data:
    # (img, expected) = data
    # result = nn.predict(img)
    # if result.argmax() != expected:
        # errors += 1

# print("errors: ", errors / 10000)

######

# nn = NeuralNetwork(2, 2, [2, 4], [Activation.SIGMOID, Activation.SOFTMAX])
# # nn = Layer(1, 2, Activation.SIGMOID, 1/1000)

# # Function to classify points based on their position in the four smaller squares
# def classify_square(x, y):
    # if 0 <= x <= 0.5 and 0 <= y <= 0.5:
        # return np.array([[1], [0], [0], [0]])  # Square 1
    # else:
        # return np.array([[0], [1], [0], [0]])  # Square 2
    # # elif 0.5 < x <= 1 and 0 <= y <= 0.5:
        # # return np.array([[0], [1], [0], [0]])  # Square 2
    # # elif 0 <= x <= 0.5 and 0.5 < y <= 1:
        # # return np.array([[0], [0], [1], [0]])  # Square 3
    # # elif 0.5 < x <= 1 and 0.5 < y <= 1:
        # # return np.array([[0], [0], [0], [1]])  # Square 4

# # Generate random points in the unit square
# num_points = 100000
# points = np.random.rand(num_points, 2)  # Shape: (100000, 2)

# # Classify points based on their position in the smaller squares
# classifications = np.array([classify_square(x, y) for x, y in points])

# points = points.reshape(num_points, 2, 1)

# for i in range(num_points):
    # # print("data:", points[i])
    # nn.fit(points[i], classifications[i])
    # # print("\n")

# num_points = 100
# points = np.random.rand(num_points, 2)  # Shape: (100000, 2)
# points = points.reshape(num_points, 2, 1)

# for i in range(num_points):
    # print(points[i])
    # print()
    # print("expected: ", classify_square(points[i][0], points[i][1]))
    # print("got:", nn.predict(points[i]))
    # print("---")

# print(nn.layers[0].weights)
# print(nn.layers[1].weights)

######

nn = NeuralNetwork(2, 2, [2, 2], [Activation.SIGMOID, Activation.SOFTMAX], 0.1)
nn.layers[0].weights = np.array([[0.1, -0.2, 0.3], [-0.4, 0.5, -0.6]])
nn.layers[1].weights = np.array([[0.15, -0.25, 0.35], [-0.45, 0.55, -0.65]])

nn.fit(np.array([[1], [0]]), np.array([[1], [0]]))
print(nn.layers[0].weights)
print(nn.layers[1].weights)
