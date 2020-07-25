#!/usr/bin/env python3
""" Compute the determinant of a simetric matrix"""


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
    for fc in range(len(matrix)):
        mat_i = matrix.copy()
        mat_i = mat_i[1:]
        mat_i = [row[0:fc] + row[fc + 1:] for row in mat_i]
        sign = (-1) ** fc
        mat_i_det = determinant(mat_i)
        deter += sign * matrix[0][fc] * mat_i_det

    return deter
