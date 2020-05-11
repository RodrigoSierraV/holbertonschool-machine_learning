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
