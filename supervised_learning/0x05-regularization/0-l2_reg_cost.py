#!/usr/bin/env python3
""" L2 Regularization """
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """ Compute cost with L2 regularization """
    sum = 0
    for k, v in weights.items():
        sum += np.sqrt(np.sum(v**2))
    return cost + lambtha/(2*m) * sum
