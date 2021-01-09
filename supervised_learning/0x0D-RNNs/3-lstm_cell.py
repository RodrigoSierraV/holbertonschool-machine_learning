#!/usr/bin/env python3
"""
Creates the class LSTMCell
"""
import numpy as np


class LSTMCell:
    """
    Class that represents an LSTM unit
    """

    def __init__(self, i, h, o):
        """
        i is the dimensionality of the data
        h is the dimensionality of the hidden state
        o is the dimensionality of the outputs
        """
        self.Wf = np.random.normal(size=(i + h, h))
        self.Wu = np.random.normal(size=(i + h, h))
        self.Wc = np.random.normal(size=(i + h, h))
        self.Wo = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """
        x_t is a numpy.ndarray of shape (m, i) that contains the data input
            for the cell
            m is the batche size for the data
        h_prev is a numpy.ndarray of shape (m, h) containing the previous
            hidden state
        c_prev is a numpy.ndarray of shape (m, h) containing the previous
            cell state
        The output of the cell should use a softmax activation function
        Returns: h_next, c_next, y
            h_next is the next hidden state
            c_next is the next cell state
            y is the output of the cell
        """
        x_concat0 = np.concatenate((h_prev, x_t), axis=1)
        f = np.matmul(x_concat0, self.Wf) + self.bf
        f = 1 / (1 + np.exp(-f))
        u = np.matmul(x_concat0, self.Wu) + self.bu
        u = 1 / (1 + np.exp(-u))
        c = np.matmul(x_concat0, self.Wc) + self.bc
        c = np.tanh(c)
        o = np.matmul(x_concat0, self.Wo) + self.bo
        o = 1 / (1 + np.exp(-o))
        c_next = (u * c) + (f * c_prev)
        h_t = o * np.tanh(c_next)
        y = np.matmul(h_t, self.Wy) + self.by
        y = np.exp(y) / np.sum(np.exp(y), axis=1, keepdims=True)
        return h_t, c_next, y
