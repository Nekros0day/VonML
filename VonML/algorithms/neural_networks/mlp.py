import numpy as np
from .activations import ACTIVATIONS


class MLP:
    """A simple multilayer perceptron."""

    def __init__(self, layer_sizes, activations, rng=None):
        assert len(layer_sizes) - 1 == len(activations), "Mismatch between layers and activations"
        rng = np.random.default_rng(rng)
        self.activations = [ACTIVATIONS[a][0] for a in activations]
        self.activation_derivs = [ACTIVATIONS[a][1] for a in activations]
        self.weights = []
        self.biases = []
        for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:]):
            self.weights.append(rng.normal(scale=0.1, size=(in_size, out_size)))
            self.biases.append(np.zeros(out_size))

    def forward(self, X):
        a = X
        self._zs = []
        self._activations = [X]
        for W, b, act in zip(self.weights, self.biases, self.activations):
            z = np.dot(a, W) + b
            a = act(z)
            self._zs.append(z)
            self._activations.append(a)
        return a

    def backward(self, grad_output, lr):
        grad = grad_output
        for i in reversed(range(len(self.weights))):
            dz = grad * self.activation_derivs[i](self._zs[i])
            a_prev = self._activations[i]
            dW = np.dot(a_prev.T, dz) / a_prev.shape[0]
            db = np.mean(dz, axis=0)
            grad = np.dot(dz, self.weights[i].T)
            self.weights[i] -= lr * dW
            self.biases[i] -= lr * db
        return grad

