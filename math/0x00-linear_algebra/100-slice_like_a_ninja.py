#!/usr/bin/env python3
"""
Contains a method that slices a matrix along a specific axes
"""


def np_slice(matrix, axes={}):
    """
    Function that slices a matrix along a specific axes
    """
    for ax, value in axes.items():
        print(matrix[value[0]:value[1]], '*******')
    return matrix[value[0]:value[1]: ax]
