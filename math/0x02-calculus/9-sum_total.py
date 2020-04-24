#!/usr/bin/env python3
""" Module to calculate the sum of the squares"""


def summation_i_squared(n):
    """ Function that calculates the sum of the squares from 1 to n"""
    if n is None or not isinstance(n, int) or n < 1:
        return
    if n == 1:
        return 1
    return n**2 + summation_i_squared(n - 1)
