#!/usr/bin/env python3
"""
Contains a method to transpose a matrix
"""


def matrix_shape(matrix):
    """
        Recursive function that finds the size of a matrix
        Return: A list of integers
    """
    if isinstance(matrix[0], list):
        return [len(matrix)] + matrix_shape(matrix[0])
    return [len(matrix)]


def matrix_transpose(matrix):
    """
        Function that transpose a matrix
        Return: a new matrix
    """
    shape_matrix = matrix_shape(matrix)
    new_matrix = [[0 for j in range(shape_matrix[0])]
                  for i in range(shape_matrix[1])]
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            new_matrix[j][i] = matrix[i][j]
    return new_matrix
