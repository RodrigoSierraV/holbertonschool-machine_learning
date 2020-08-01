#!/usr/bin/env python3
""" calculates the marginal probability of obtaining the data """
import numpy as np


def posterior(x, n, P, Pr):
    """
    x is the number of patients that develop severe side effects
    n is the total number of patients observed
    P is a 1D numpy.ndarray containing the various hypothetical probabilities
        of patients developing severe side effects
    Pr is a 1D numpy.ndarray containing the prior beliefs about P
    """
    if not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    if type(x) != int or x < 0:
        mg = "x must be an integer that is greater than or equal to 0"
        raise ValueError(mg)
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if not isinstance(Pr, np.ndarray) or P.shape != Pr.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    if np.amin(P) < 0 or np.amax(P) > 1:
        raise ValueError("All values in P must be in the range [0, 1]")
    if np.amin(Pr) < 0 or np.amax(Pr) > 1:
        raise ValueError("All values in Pr must be in the range [0, 1]")
    if not np.isclose([np.sum(Pr)], [1])[0]:
        raise ValueError("Pr must sum to 1")

    n_factorial = np.math.factorial(n)
    den = np.math.factorial(x) * np.math.factorial(n - x)
    likelihood = n_factorial / den * (P ** x) * ((1 - P) ** (n - x))
    inter = likelihood * Pr
    marginal = np.sum(inter)
    return inter / marginal
