#!/usr/bin/env python3
""" Module to Create a neuron
"""
import numpy as np


class Neuron:
    """ Class that models a Neuron"""
    def __init__(self, nx):
        """ Class constructor for Neuron"""
        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        self.nx = nx
        self.__W = np.random.normal(size=(1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """ Getter for weights"""
        return self.__W

    @property
    def b(self):
        """ Getter for bias"""
        return self.__b

    @property
    def A(self):
        """ Getter for activated output"""
        return self.__A

    def forward_prop(self, X):
        """ Calculates the forward propagation of the neuron"""
        self.__A = np.matmul(self.__W, X) + self.__b

        def sigmoid(num):
            """ Logit or sigmoid function"""
            return 1/(1 + np.e**(-num))
        self.__A = np.apply_along_axis(sigmoid, 1, self.__A)
        return self.__A

    def cost(self, Y, A):
        """ Calculates the cost of the model using logistic regression"""
        p1 = np.matmul(Y, np.log(A).transpose())
        p0 = np.matmul((1 - Y), np.log(1.0000001 - A).transpose())
        cost = -1 / Y.shape[1] * (p1 + p0)
        return cost[0][0]
