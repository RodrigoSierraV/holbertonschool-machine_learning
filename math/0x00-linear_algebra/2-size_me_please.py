#!/usr/bin/env python3
def matrix_shape(matrix):
    """
        Recursive function that finds the size of a matrix
        Return: as a list of integers
    """
    if isinstance(matrix[0], list):
        return [len(matrix)] + matrix_shape(matrix[0])
    return [len(matrix)]
