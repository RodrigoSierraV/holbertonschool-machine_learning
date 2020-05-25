#!/usr/bin/env python3
""" Module to shuffle a matrix"""
import numpy as np


def shuffle_data(X, Y):
    """Shuffles a matrix of data"""
    data = np.hstack((X, Y))
    data = np.random.permutation(data)
    return data[:, : -Y.shape[1]], data[:, -Y.shape[1]:]
