#!/usr/bin/env python3
"""
Contains a method that adds two matrices element-wise
"""


def matrix_shape(matrix):
    """
        Recursive function that finds the size of a matrix
        Return: A list of integers
    """
    if isinstance(matrix[0], list):
        return [len(matrix)] + matrix_shape(matrix[0])
    return [len(matrix)]


def add_matrices2D(mat1, mat2):
    """
        Function adds two matrices element-wise
        Return: A list of integers
    """
    if matrix_shape(mat1) != matrix_shape(mat2):
        return
    return [[mat1[i][j] + mat2[i][j] for j in range(len(mat1[i]))]
            for i in range(len(mat1))]
