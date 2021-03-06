#!/usr/bin/env python3
""" initializes cluster centroids for K-means """
import numpy as np


def initialize(X, k):
    """
    X is a numpy.ndarray of shape (n, d) containing the dataset that will be
        used for K-means clustering
        n is the number of data points
        d is the number of dimensions for each data point
    k is a positive integer containing the number of clusters
    The cluster centroids should be initialized with a multivariate uniform
        distribution along each dimension in d:
    The minimum values for the distribution should be the minimum values of
        X along each dimension in d
    The maximum values for the distribution should be the maximum values of
        X along each dimension in d
    Returns: a numpy.ndarray of shape (k, d) containing the initialized
        centroids for each cluster, or None on failure
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(k, int) or k <= 0 or k >= X.shape[0]:
        return None

    min_x = np.amin(X, axis=0)
    max_x = np.amax(X, axis=0)
    d = X.shape[1]
    return np.random.uniform(min_x, max_x, (k, d))
