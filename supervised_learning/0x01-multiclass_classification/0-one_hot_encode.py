#!/usr/bin/env python3
"""Module to create a one-hot encoder"""

import numpy as np


def one_hot_encode(Y, classes):
    """Encodes array Y"""
    try:
        out = np.zeros((classes, Y.shape[0]))
        out[Y, np.arange(Y.shape[0])] = 1
    except Exception:
        return None
    return out
