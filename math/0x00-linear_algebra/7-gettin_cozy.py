#!/usr/bin/env python3
"""
Contains a method that concatenates two matrices along a specific axis
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
        Function that concatenates two matrices along a specific axis
        Return: A new matrix
    """
    mat11 = [[i for i in j] for j in mat1]
    mat22 = [[i for i in j] for j in mat2]
    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return
        return mat11 + mat22
    if axis == 1:
        if len(mat1) != len(mat2):
            return
        return [mat11[i] + mat22[i] for i in range(len(mat11))]
