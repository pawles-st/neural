from itertools import islice
from neural import Activation, Layer, NeuralNetwork
import numpy as np
import mnist_loader
import random

(training_data, validation_data, test_data) = mnist_loader.load_data_wrapper()

nn = NeuralNetwork(784, 3, [100, 100, 10], [Activation.SIGMOID, Activation.SIGMOID, Activation.SOFTMAX], learning_rate = 0.01)

training_data = list(training_data)

def chunked(iterable, chunk_size):
    it = iter(iterable)
    while True:
        chunk = list(islice(it, chunk_size))
        if not chunk:
            break
        yield chunk

for i in range(10):
    random.shuffle(training_data)
    print("epoch", i + 1)
    for data in chunked(training_data, 10):
        images = np.hstack([d[0] for d in data])
        expected = np.hstack([d[1] for d in data])
        nn.fit(images, expected)

errors = 0

for data in test_data:
    (img, expected) = data
    result = nn.predict(img)
    if result.argmax() != expected:
        errors += 1

print("errors: ", errors / 10000)
