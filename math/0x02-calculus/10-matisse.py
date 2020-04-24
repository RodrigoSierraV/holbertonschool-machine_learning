#!/usr/bin/env python3
""" Module that calculates the derivative of a polynomial"""


def poly_derivative(poly):
    """calculates the derivative of a polynomial"""
    verify = all([isinstance(i, int) for i in poly])
    if verify is False:
        return
    derivative = [poly[coef] * coef for coef in range(1, len(poly))]
    if sum(derivative) == 0:
        return [0]
    return derivative
