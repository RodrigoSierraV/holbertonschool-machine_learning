#!/usr/bin/env python3
"""
Contains a method that adds two arrays
"""


matrix_shape = __import__('2-size_me_please').matrix_shape


def add_arrays(arr1, arr2):
    """
        Function that adds two arrays
        Return: A new array
    """
    if matrix_shape(arr1) != matrix_shape(arr2):
        return
    return [arr1[i] + arr2[i] for i in range(len(arr1))]
