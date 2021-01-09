#!/usr/bin/env python3
"""
Create the class BidirectionalCell
"""
import numpy as np


class BidirectionalCell:
    """
    Class that represents a bidirectional cell of an RNN
    """

    def __init__(self, i, h, o):
        """
        i is the dimensionality of the data
        h is the dimensionality of the hidden states
        o is the dimensionality of the outputs
        """
        self.Whf = np.random.normal(size=(i + h, h))
        self.Whb = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(2 * h, o))
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        x_t is a numpy.ndarray of shape (m, i) that contains the data input
            for the cell
            m is the batch size for the data
        h_prev is a numpy.ndarray of shape (m, h) containing the previous
            hidden state
        Returns: h_next, the next hidden state
        """
        x_concat = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.matmul(x_concat, self.Whf) + self.bhf
        h_next = np.tanh(h_next)
        return h_next
