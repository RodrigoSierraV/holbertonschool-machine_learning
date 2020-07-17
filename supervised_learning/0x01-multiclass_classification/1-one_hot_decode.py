#!/usr/bin/env python3
""" One-hot decoder """
import numpy as np


def one_hot_decode(one_hot):
    """Decodes a one-hot encode"""
    if type(one_hot) is not np.ndarray:
        return None
    if len(one_hot) == 0 or len(one_hot.shape) != 2:
        return None
    return np.argmax(one_hot, axis=0)

