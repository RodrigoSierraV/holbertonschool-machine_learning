#!/usr/bin/env python3
""" Compute the determinant of a simetric matrix"""


def getMatrixMinor(m, i, j):
    """calculates the minor of a squared matrix"""
    return [row[:j] + row[j+1:] for row in (m[:i] + m[i+1:])]


def determinant(matrix):
    """
    matrix is a list of lists whose determinant should be calculated
    Returns: the determinant of matrix
    """
    if not isinstance(matrix, list):
        raise TypeError('matrix must be a list of lists')
    list_of_lists = [isinstance(row, list) for row in matrix]
    square = [len(row) == len(matrix) for row in matrix]
    if len(matrix) == 1 and len(matrix[0]) == 0:
        return 1
    if len(matrix) == 1 and len(matrix[0]) == 1:
        return matrix[0][0]
    if not all(list_of_lists) or len(matrix) == 0:
        raise TypeError('matrix must be a list of lists')
    if not all(square):
        raise ValueError('matrix must be a square matrix')
    if len(matrix) is 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    deter = 0
    for c in range(len(matrix)):
        deter += ((-1.0) ** c) * matrix[0][c] *\
                 determinant(getMatrixMinor(matrix, 0, c))
    return int(deter)
