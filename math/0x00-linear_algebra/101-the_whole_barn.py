#!/usr/bin/env python3
"""
Contains methods that adds two matrices
"""


def matrix_shape(matrix):
    """
        Recursive function that finds the size of a matrix
        Return: A list of integers
    """
    if isinstance(matrix[0], list):
        return [len(matrix)] + matrix_shape(matrix[0])
    return [len(matrix)]


def mat_add(mat1, mat2):
    """
        Recursive function that adds two matrices
        Return: A list of integers
    """
    if isinstance(mat1[0], list):
        return [mat_add(mat1[i], mat2[i]) for i in range(len(mat1))]
    z = zip(mat1, mat2)
    return list(map(lambda i: i[0] + i[1], z))


def add_matrices(mat1, mat2):
    """
    Function that adds two matrices
    """
    if matrix_shape(mat1) != matrix_shape(mat2):
        return
    return mat_add(mat1, mat2)
