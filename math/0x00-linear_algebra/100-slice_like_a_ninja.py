#!/usr/bin/env python3
"""
Contains a method that slices a matrix along specific axes
"""


def np_slice(matrix, axes={}):
    """
    Function that slices a matrix along specific axes
    """
    l = []
    for i in range(len(matrix.shape)):
        if i not in axes:
            l.append(slice(None, None, None))
            continue
        l.append(slice(*axes[i]))
    return matrix[tuple(l)]
