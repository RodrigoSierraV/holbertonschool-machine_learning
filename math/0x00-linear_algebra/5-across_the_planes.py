#!/usr/bin/env python3
"""
Contains a method that adds two matrices element-wise
"""


def add_matrices2D(mat1, mat2):
    """
        Function adds two matrices element-wise
        Return: A new matrix
    """
    verify_len1 = len(mat1) == len(mat2)
    if verify_len1:
        verify_len2 = all([len(mat1[i]) == len(mat2[i])
                           for i in range(len(mat1))])
        if verify_len2:
            return [[mat1[i][j] + mat2[i][j] for j in range(len(mat1[i]))]
                    for i in range(len(mat1))]
    return
