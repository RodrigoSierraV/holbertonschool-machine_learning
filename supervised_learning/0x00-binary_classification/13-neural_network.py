#!/usr/bin/env python3
""" Module to Create a neural network
"""
import numpy as np


class NeuralNetwork:
    """ Class that models a Neural network"""
    def __init__(self, nx, nodes):
        """ Class constructor for Neural network"""
        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if not isinstance(nodes, int):
            raise TypeError('nodes must be an integer')
        if nodes < 1:
            raise ValueError('nodes must be a positive integer')
        self.__nx = nx
        self.__W1 = np.random.normal(size=(nodes, nx))
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.normal(size=(1, nodes))
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """ Getter for weights"""
        return self.__W1

    @property
    def b1(self):
        """ Getter for bias"""
        return self.__b1

    @property
    def A1(self):
        """ Getter for activated output"""
        return self.__A1

    @property
    def W2(self):
        """ Getter for weights"""
        return self.__W2

    @property
    def b2(self):
        """ Getter for bias"""
        return self.__b2

    @property
    def A2(self):
        """ Getter for activated output"""
        return self.__A2

    def forward_prop(self, X):
        """ Calculates the forward propagation of the neuron"""

        def sigmoid(num):
            """ Logit or sigmoid function"""
            return 1/(1 + np.e**(-num))
        self.__A1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = sigmoid(self.__A1)
        self.__A2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = sigmoid(self.__A2)

        return self.__A1, self.__A2

    def cost(self, Y, A):
        """ Calculates the cost of the model using logistic regression"""
        p1 = np.matmul(Y, np.log(A).transpose())
        p0 = np.matmul((1 - Y), np.log(1.0000001 - A).transpose())
        cost = -1 / Y.shape[1] * (p1 + p0)
        return cost[0][0]

    def evaluate(self, X, Y):
        """ Evaluates the neuron’s predictions"""
        A1, A2 = self.forward_prop(X)
        cost = self.cost(Y, A2)
        A2 = np.where(A2 >= 0.5, 1, 0)

        return A2, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """ Calculates one pass of gradient descent on the neuron"""
        dz2 = A2 - Y
        dw2 = 1/X.shape[1] * np.matmul(dz2, A1.T)
        db2 = 1/X.shape[1] * np.sum(dz2, axis=1, keepdims=True)
        dz1 = np.matmul(self.__W2.T, dz2) * A1 * (1 - A1)
        dw1 = 1 / X.shape[1] * np.matmul(dz1, X.T)
        db1 = 1/X.shape[1] * np.sum(dz1, axis=1, keepdims=True)
        self.__W1 = self.__W1 - alpha * dw1
        self.__b1 = self.__b1 - alpha * db1
        self.__W2 = self.__W2 - alpha * dw2
        self.__b2 = self.__b2 - alpha * db2
