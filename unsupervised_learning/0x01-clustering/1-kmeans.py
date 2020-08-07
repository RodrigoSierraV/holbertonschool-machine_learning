#!/usr/bin/env python3
""" performs K-means on a dataset """
import numpy as np


def kmeans(X, k, iterations=1000):
    """
    X is a numpy.ndarray of shape (n, d) containing the dataset
        n is the number of data points
        d is the number of dimensions for each data point
    k is a positive integer containing the number of clusters
    iterations is a positive integer containing the maximum number of
        iterations that should be performed
    If no change in the cluster centroids occurs between iterations, your
        function should return
    Initialize the cluster centroids using a multivariate uniform distribution
        (based on0-initialize.py)
    If a cluster contains no data points during the update step, reinitialize
        its centroid
    Returns: C, clss, or None, None on failure
        C is a numpy.ndarray of shape (k, d) containing the centroid means for
            each cluster
        clss is a numpy.ndarray of shape (n,) containing the index of the
            cluster in C that each data point belongs to
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(k, int) or k < 1:
        return None, None
    if not isinstance(iterations, int) or iterations < 1:
        return None, None

    n, d = X.shape
    if n < k:
        return None, None

    min_x = np.amin(X, axis=0)
    max_x = np.amax(X, axis=0)
    init_centers = np.random.uniform(min_x, max_x, size=(k, d))

    for i in range(iterations):
        prev_centers = np.ndarray.copy(init_centers)
        deltas = X - init_centers[:, np.newaxis]
        distance = np.sqrt((deltas ** 2).sum(axis=2))
        clss = np.argmin(distance, axis=0)
        for j in range(k):
            if len(X[clss == j]) == 0:
                init_centers[j] = np.random.uniform(np.min(X, axis=0),
                                                    np.max(X, axis=0),
                                                    size=(1, d))
            else:
                init_centers[j] = (X[clss == j]).mean(axis=0)
        deltas = X - init_centers[:, np.newaxis]
        distance = np.sqrt((deltas ** 2).sum(axis=2))
        clss = np.argmin(distance, axis=0)
        if np.all(prev_centers == init_centers):
            return init_centers, clss

    return init_centers, clss
