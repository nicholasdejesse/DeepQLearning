import math
import numpy as np
import random

class Network:
    def __init__(self, layers: tuple):
        self.layers = layers
        self.weights = []
        self.bias = []
        self.gradient_clip = 1000
        self.epochs_trained = 0
        for i in range(len(layers) - 1):
            w = np.random.normal(0, math.sqrt(2 / layers[i]), (layers[i], layers[i+1]))
            b = np.random.normal(0, math.sqrt(2 / layers[i]), (1, layers[i+1]))
            self.weights.append(w)
            self.bias.append(b)

    def copy_weights_and_biases(self):
        weight_copy = [np.copy(w) for w in self.weights]
        bias_copy = [np.copy(b) for b in self.bias]
        return weight_copy, bias_copy

    def load_weights_and_biases(self, weights_and_biases):
        for i, w in enumerate(weights_and_biases[0]):
            self.weights[i] = w
        for j, b in enumerate(weights_and_biases[1]):
            self.bias[j] = b

    def result(self, x):
        return self._forward_pass(np.reshape(np.array(np.copy(x)), (1, -1)))[-1][0]

    def _forward_pass(self, x):
        res = []
        if len(x.shape) == 0:
            x = np.reshape(np.copy(x), (1, -1))
        else:
            n = x.shape[0]
            x = np.reshape(np.copy(x), (n, -1))
        res.append(x)
        for i in range(len(self.weights)):
            x = np.dot(x, self.weights[i])
            # During batch training, the bias is added to each row
            x = x + self.bias[i]
            if i < len(self.weights) - 1:
                x = self.relu(x)
            res.append(x)
        return res

    def train(self, x, y, batch_size, epochs, learning_rate, loss_at_epochs_debug_list = None):
        if batch_size > len(x):
            raise Exception("Batch size greater than size of training data")
        x, y = self.clean(x, y)

        for e in range(epochs):
            self.epochs_trained += 1
            sample_indeces = random.sample(range(0, len(x)), batch_size)
            x_samples = np.array([x[i] for i in sample_indeces])
            y_samples = np.reshape(np.array([y[i] for i in sample_indeces]), (self.layers[-1], -1)).T

            result = self._forward_pass(x_samples)
            loss = self.loss(result[-1], y_samples) / batch_size
            loss_deriv = self.loss_deriv(result[-1], y_samples) / batch_size

            if loss_at_epochs_debug_list is not None:
                if self.epochs_trained % 5 == 0:
                    loss_at_epochs_debug_list.append(loss)

            # Traverse weights from right to left
            for i in range(len(self.weights) - 1, -1, -1):
                weight_deriv = np.dot(result[i].T, loss_deriv).mean(axis=1, keepdims=True)
                bias_deriv = loss_deriv.sum(axis=0, keepdims=True)

                weight_deriv = np.clip(weight_deriv, -1 * self.gradient_clip, self.gradient_clip)
                bias_deriv = np.clip(bias_deriv, -1 * self.gradient_clip, self.gradient_clip)

                self.weights[i] -= learning_rate * weight_deriv
                self.bias[i] -= learning_rate * bias_deriv

                loss_deriv = np.dot(loss_deriv, self.weights[i].T)
                loss_deriv = np.multiply(loss_deriv, self.relu_deriv(result[i]))

    def loss(self, x, target):
        return np.sum(0.5 * np.power(x - target, 2))

    def loss_deriv(self, x, target):
        return x - target

    # Not used right now but may be in the future
    # def sigmoid(self, x):
    #     return 1 / (1 + np.exp(-x))
    
    # def sigmoid_deriv(self, x):
    #     sig = self.sigmoid(x)
    #     return sig * (1 - sig)

    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_deriv(self, x):
        return np.where(x > 0, 1, 0)
    
    def clean(self, x, y):
        clean_x = [np.array(i) for i in x]
        clean_y = [np.array(i) for i in y]
        return clean_x, clean_y