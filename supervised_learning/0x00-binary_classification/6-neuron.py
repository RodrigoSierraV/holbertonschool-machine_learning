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

    def evaluate(self, X, Y):
        """ Evaluates the neuron’s predictions"""
        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        A = np.where(A >= 0.5, 1, 0)

        return A, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """ Calculates one pass of gradient descent on the neuron"""
        gradient = np.matmul(X, (A - Y).T) / X.shape[1]
        self.__W = self.__W - alpha * gradient.transpose()
        db = np.sum(A - Y) / X.shape[1]
        self.__b = self.__b - alpha * db
        return self.__W, self.__b

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """ Trains the neuron"""
        if not isinstance(iterations, int):
            raise TypeError('iterations must be an integer')
        if iterations < 0:
            raise ValueError('iterations must be a positive integer')
        if not isinstance(alpha, float):
            raise TypeError('alpha must be a float')
        if alpha < 0:
            raise ValueError('alpha must be positive')

        for iteration in range(iterations):
            self.__A, cost = self.evaluate(X, Y)
            self.__W, self.__b = self.gradient_descent(X, Y, self.__A, alpha)

        return self.evaluate(X, Y)
