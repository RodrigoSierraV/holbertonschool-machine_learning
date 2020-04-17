#!/usr/bin/env python3
"""
Contains a method that slices a matrix along a specific axes
"""


def matrix_shape(matrix):
    """
        Recursive function that finds the size of a matrix
        Return: A list of integers
    """
    if isinstance(matrix[0], list):
        return [len(matrix)] + matrix_shape(matrix[0])
    return [len(matrix)]


def cat_matrices(mat1, mat2, axis=0):
    """
        Function that concatenates two matrices along a specific axis
        Return: A new matrix
    """
    shape1 = matrix_shape(mat1)
    shape2 = matrix_shape(mat2)
    dim1 = len(shape1)
    dim2 = len(shape2)
    if axis == 0:
        if (dim1 == 1 and dim2 == 1):
            if dim1 == dim2:
                return mat1 + mat2
            else:
                return
        if shape1[1:] == shape2[1:]:
            return mat1 + mat2
        return
    if axis == 1:
        if dim1 == 1 and dim2 == 1:
            return mat1 + mat2
        if dim1 == 2 and dim2 == 2:
            if shape1[0] == shape2[0]:
                return [mat1[i] + mat2[i] for i in range(len(mat1))]
            return
        if (shape1[0] == shape2[0]) and (shape1[2:] == shape2[2:]):
            return [mat1[i] + mat2[i] for i in range(len(mat1))]
        return
