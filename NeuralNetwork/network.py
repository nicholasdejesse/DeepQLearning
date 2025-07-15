import math
import numpy as np
import random

class Network:
    def __init__(self, layers: tuple):
        self.layers = layers
        self.weights = []
        self.bias = []
        for i in range(len(layers) - 1):
            # TODO: Check for bug in initialization since exploding gradients happen quite often
            w = np.random.normal(0, math.sqrt(2 / layers[i]), (layers[i], layers[i+1]))
            b = np.random.normal(0, math.sqrt(2 / layers[i]), (1, layers[i+1]))
            self.weights.append(w)
            self.bias.append(b)

    def result(self, x):
        return self._forward_pass(np.reshape(np.array(x), (1, -1)))[-1][0]

    def _forward_pass(self, x):
        res = []
        if len(x.shape) == 0:
            x = np.reshape(x, (1, -1))
        else:
            n = x.shape[0]
            x = np.reshape(x, (n, -1))
        res.append(x)
        for i in range(len(self.weights)):
            x = np.dot(x, self.weights[i])
            # During batch training, the bias is added to each row
            x = x + self.bias[i]
            if i < len(self.weights) - 1:
                x = self.relu(x)
            # print(x)
            res.append(x)
        return res

    def train(self, x, y, batch_size, epochs, learning_rate, loss_at_epochs_debug_list = None):
        if batch_size > len(x):
            raise Exception("Batch size greater than size of training data")
        x, y = self.clean(x, y)
        for _ in range(epochs):
            sample_indeces = random.sample(range(0, len(x)), batch_size)
            x_samples = np.array([x[i] for i in sample_indeces])
            y_samples = np.reshape(np.array([y[i] for i in sample_indeces]), (self.layers[-1], -1)).T
            result = self._forward_pass(x_samples)
            loss = self.loss(result[-1], y_samples) / batch_size
            loss_deriv = self.loss_deriv(result[-1], y_samples) / batch_size

            # print(loss_deriv)

            if loss_at_epochs_debug_list is not None:
                loss_at_epochs_debug_list.append(loss)

            # Traverse weights from right to left
            for i in range(len(self.weights) - 1, -1, -1):
                weight_deriv = np.dot(result[i].T, loss_deriv).mean(axis=1, keepdims=True)
                bias_deriv = loss_deriv.sum(axis=0, keepdims=True)

                self.weights[i] -= learning_rate * weight_deriv
                self.bias[i] -= learning_rate * bias_deriv

                loss_deriv = np.dot(loss_deriv, self.weights[i].T)
                loss_deriv = np.multiply(loss_deriv, self.relu_deriv(result[i]))

    def loss(self, x, target):
        return np.sum(0.5 * np.power(x - target, 2))

    def loss_deriv(self, x, target):
        return x - target

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