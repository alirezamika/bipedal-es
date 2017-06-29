import numpy as np


class Model(object):

    def __init__(self):
        self.weights = [np.zeros(shape=(24, 16)), np.zeros(shape=(16, 16)), np.zeros(shape=(16, 4))]

    def predict(self, inp):
        out = np.expand_dims(inp.flatten(), 0)
        out = out / np.linalg.norm(out)
        for layer in self.weights:
            out = np.dot(out, layer)
        return out[0]

    def get_weights(self):
        return self.weights

    def set_weights(self, weights):
        self.weights = weights
