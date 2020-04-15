#!/usr/bin/env python3
"""
Contains a method to find the shape of a matrix
"""


def matrix_shape(matrix):
    """
        Recursive function that finds the size of a matrix
        Return: A list of integers
    """
    if isinstance(matrix[0], list):
        return [len(matrix)] + matrix_shape(matrix[0])
    return [len(matrix)]
