from neural import NeuralNetwork
import numpy as np
import mnist_loader

(training_data, validation_data, test_data) = mnist_loader.load_data_wrapper()

nn = NeuralNetwork(784, 4, [1000, 2000, 1000, 10])

for (i, data) in enumerate(training_data):
    if i % 1000 == 0:
        print(i)
    (img, expected) = data
    nn.fit(img / 255, expected)

errors = 0

for data in test_data:
    (img, expected) = data
    result = nn.predict(img / 255)
    print(result, expected)
    if np.linalg.norm(result - expected) > 0.1:
        errors += 1

print("errors: ", errors / 10000)

