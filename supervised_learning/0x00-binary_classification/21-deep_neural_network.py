#!/usr/bin/env python3
""" Module to create a deep neural network """

import numpy as np


class DeepNeuralNetwork:
    """ Class that models a deep neural
        network for binary classification"""

    def __init__(self, nx, layers):
        """ Class constructor """
        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if not isinstance(layers, list):
            raise TypeError('layers must be a list of positive integers')
        if len(layers) == 0:
            raise TypeError('layers must be a list of positive integers')
        self.__L = len(layers)
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
            self.__cache[A] = 1 / (1 + np.exp(-z))
        out = 'A{}'.format(self.__L)
        return self.__cache[out], self.__cache

    def cost(self, Y, A):
        """Compute the cost with logistic regression"""
        loss_one = np.matmul(Y, np.log(A).T)
        loss_two = np.matmul(1 - Y, np.log(1.0000001 - A.T))
        return np.sum(-(loss_one + loss_two)) / Y.shape[1]

    def evaluate(self, X, Y):
        """Evaluate predictions"""
        A, cache = self.forward_prop(X)
        cost = self.cost(Y, A)
        return np.where(A >= 0.5, 1, 0), cost

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
