#!/usr/bin/env python3
"""
Contains a method that performs element-wise
addition, subtraction, multiplication, and division
"""


def np_elementwise(mat1, mat2):
    """
    Function that performs element-wise
    addition, subtraction, multiplication, and division
    """
    new = mat1 + mat2
    return (mat1 + mat2, mat1 - mat2,
            mat1 * mat2, mat1 / mat2)
