#!/usr/bin/env python3
"""
Contains a method that performs matrix multiplication
"""


def mul_arrays(arr1, arr2):
    """
        Function that calculates the linear combination of two arrays
    """
    z = zip(arr1, arr2)
    sum = 0
    for a, b in z:
        sum += a * b
    return sum


def mat_mul(mat1, mat2):
    """
        Function that performs matrix multiplication
        Return: A new matrix
    """
    if len(mat1[0]) != len(mat2):
        return
    mat1 = [[i for i in j] for j in mat1]
    mat2 = [[i for i in j] for j in mat2]
    mat_mul = [[0 for i in range(len(mat2[0]))] for j in range(len(mat1))]

    for i in range(len(mat_mul)):
        for j in range(len(mat_mul[i])):
            mat_mul[i][j] = mul_arrays(mat1[i], [k[j] for k in mat2])
    return mat_mul
