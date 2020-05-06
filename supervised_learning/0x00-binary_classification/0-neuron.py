#!/usr/bin/env python3
import numpy as np
""" Module to Create a neuron
"""


class Neuron:
    """ Class that models a Neuron"""
    def __init__(self, nx):
        """ Class constructor for Neuron"""
        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive')
        self.nx = nx
        self.W = np.random.normal(size=(1, nx))
        self.b = 0
        self.A = 0
