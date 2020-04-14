#!/usr/bin/env python3
def matrix_shape(matrix):
    if isinstance(matrix[0], list):
        return [len(matrix)] + matrix_shape(matrix[0])
    return [len(matrix)]