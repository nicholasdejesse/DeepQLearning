import math
import numpy as np
import random

class Network:
    def __init__(self, layers: tuple):
        self.layers = layers
        self.weights = []
        self.bias = []
        self.gradient_clip = math.inf
        for i in range(len(layers) - 1):
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
            res.append(x)
        return res

    def train(self, x, y, batch_size, epochs, learning_rate, actions_to_focus = None, loss_at_epochs_debug_list = None):
        if batch_size > len(x):
            raise Exception("Batch size greater than size of training data")
        x, y = self.clean(x, y)
        for _ in range(epochs):
            sample_indeces = random.sample(range(0, len(x)), batch_size)
            x_samples = np.array([x[i] for i in sample_indeces])
            y_samples = np.reshape(np.array([y[i] for i in sample_indeces]), (self.layers[-1], -1)).T

            actions = None
            if actions_to_focus is not None:
                actions = [actions_to_focus[i] for i in sample_indeces]

            result = self._forward_pass(x_samples)
            loss = self.loss(result[-1], y_samples, actions) / batch_size
            loss_deriv = self.loss_deriv(result[-1], y_samples, actions) / batch_size

            if loss_at_epochs_debug_list is not None:
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

    def loss(self, x, target, actions):
        mse = 0.5 * np.power(x - target, 2)
        if actions is not None:
            for i, c in enumerate(mse):
                keep = c[actions[i]]
                c.fill(0)
                c[actions[i]] = keep
        return np.sum(mse)

    def loss_deriv(self, x, target, actions):
        deriv = x - target
        if actions is not None:
            for i, c in enumerate(deriv):
                keep = c[actions[i]]
                c.fill(0)
                c[actions[i]] = keep
        return deriv

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