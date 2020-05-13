#!/usr/bin/env python3
""" Module to create a deep neural network """

import numpy as np

import pickle

import matplotlib.pyplot as plt


class DeepNeuralNetwork:
    """ Class that models a deep neural
        network for binary classification"""

    def __init__(self, nx, layers, activation='sig'):
        """ Class constructor """
        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if not isinstance(layers, list):
            raise TypeError('layers must be a list of positive integers')
        if len(layers) == 0:
            raise TypeError('layers must be a list of positive integers')
        if activation not in ['sig', 'tanh']:
            raise ValueError("activation must be 'sig' or 'tanh'")
        self.__L = len(layers)
        self.__activation = activation
        self.__cache = {}
        self.__weights = {}
        l_prev = nx
        for layer in range(len(layers)):
            if not isinstance(layers[layer], int) or layers[layer] <= 0:
                raise TypeError('layers must be a list of positive integers')
            key = 'W{}'.format(layer + 1)
            bias = 'b{}'.format(layer + 1)
            self.__weights[key] = np.random.randn(layers[layer], l_prev)\
                * np.sqrt(2 / l_prev)
            self.__weights[bias] = np.zeros((layers[layer], 1))
            l_prev = layers[layer]

    @property
    def L(self):
        """ Return layers in the neural network"""
        return self.__L

    @property
    def activation(self):
        """ activation function for hidden layers """
        return self.__activation

    @property
    def cache(self):
        """ Return intermediary values of the network"""
        return self.__cache

    @property
    def weights(self):
        """Return weights and bias of the network"""
        return self.__weights

    def forward_prop(self, X):
        """Forward propagation"""
        self.__cache['A0'] = X
        for layer in range(self.__L):
            key = 'W{}'.format(layer + 1)
            bias = 'b{}'.format(layer + 1)
            weights = self.__weights[key]
            cache = 'A{}'.format(layer)
            cache = self.__cache[cache]
            z = np.matmul(weights, cache) + self.__weights[bias]
            A = 'A{}'.format(layer + 1)
            if layer < self.__L - 1:
                if self.__activation == 'sig':
                    self.__cache[A] = 1 / (1 + np.exp(-z))
                else:
                    self.__cache[A] = np.tanh(z)
            else:
                prop = np.sum(np.exp(z), axis=0, keepdims=True)
                self.__cache[A] = np.exp(z) / prop
        out = 'A{}'.format(self.__L)
        return self.__cache[out], self.__cache

    def cost(self, Y, A):
        """Compute the cost with softmax"""
        return -np.sum(Y * np.log(A)) / Y.shape[1]

    def evaluate(self, X, Y):
        """Evaluate predictions"""
        A, cache = self.forward_prop(X)
        probs = np.amax(A, axis=0, keepdims=True)
        return \
            np.where(self.__cache['A'+str(self.__L)] == probs, 1, 0),\
            self.cost(Y, A)

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Compute one pass of gradient descent
           on the neural network"""
        weights = self.weights.copy()
        for layer in range(self.__L, 0, -1):
            key = 'A{}'.format(layer)
            input_layer = 'A{}'.format(layer - 1)
            w = 'W{}'.format(layer)
            out = 'A{}'.format(self.__L)
            bias = 'b{}'.format(layer)
            if layer == self.__L:
                dz = cache[out] - Y
                dw = np.matmul(dz, cache[input_layer].T) / Y.shape[1]
            else:
                w1 = 'W{}'.format(layer + 1)
                back = np.matmul(weights[w1].T, dz)
                derivative = cache[key] * (1 - cache[key])
                dz = back * derivative
                dw = 1 / Y.shape[1] * np.matmul(dz, cache[input_layer].T)
            db = 1 / Y.shape[1] * np.sum(dz, axis=1, keepdims=True)
            self.__weights[w] = weights[w] - alpha * dw
            self.__weights[bias] = weights[bias] - alpha * db
        return self.__weights

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """Train the deep neural network"""
        if not isinstance(iterations, int):
            raise TypeError('iterations must be an integer')
        if iterations < 1:
            raise ValueError('iterations must be a positive integer')
        if not isinstance(alpha, float):
            raise TypeError('alpha must be a float')
        if alpha < 0:
            raise ValueError('alpha must be positive')
        to_print = []
        for iteration in range(iterations):
            _, self.__cache = self.forward_prop(X)
            key = "A{}".format(self.__L)
            cost = self.cost(Y, self.__cache[key])
            if iteration % step == 0 or iteration == iterations:
                to_print.append((iteration, cost))
                if verbose:
                    print("Cost after {} iterations: {}".
                          format(iteration, cost))
            self.gradient_descent(Y, self.__cache, alpha)
        if verbose:
            print("Cost after {} iterations: {}".
                  format(iterations, cost))
        if graph:
            plt.plot(*zip(*to_print), 'b-')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()

        return self.evaluate(X, Y)

    def save(self, filename):
        """Method to serialize model"""
        if filename[-4:] != ".pkl":
            filename = filename + ".pkl"
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """method to deserialize a model from a pickle file"""
        try:
            with open(filename, 'rb') as f:
                obj = pickle.load(f)
            return obj
        except FileNotFoundError:
            return None
