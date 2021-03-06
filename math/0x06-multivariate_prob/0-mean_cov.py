#!/usr/bin/env python3
""" calculates the mean and covariance of a data set """
import numpy as np


def mean_cov(X):
    """
    X is a numpy.ndarray of shape (n, d) containing the data set:
    n is the number of data points
    d is the number of dimensions in each data point
    Returns: mean, cov:
        mean numpy.ndarray of shape (1, d) containing the mean of the data set
        cov numpy.ndarray of shape (d, d) containing the covariance matrix
            of the data set
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        raise TypeError("X must be a 2D numpy.ndarray")
    if X.shape[0] < 2:
        raise ValueError("X must contain multiple data points")
    d = X.shape[1]
    mean = np.mean(X, axis=0)
    a = X - mean
    c = np.matmul(a.T, a)
    cov = c/(X.shape[0] - 1)
    return mean.reshape((1, d)), cov
