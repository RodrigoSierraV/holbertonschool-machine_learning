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
        """ Return hold all intermediary values of the network"""
        return self.__cache

    @property
    def weights(self):
        """Return hold all weights and biased of the network"""
        return self.__weights
