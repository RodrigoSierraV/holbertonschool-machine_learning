#!/usr/bin/env python3
""" Module that calculates the integral of a polynomial"""


def poly_integral(poly, C=0):
    """calculates the integral of a polynomial"""
    verify = all([isinstance(i, int) for i in poly])
    if verify is False:
        return
    integral = [poly[coef] // (coef + 1) if poly[coef] % (coef + 1) == 0
                else poly[coef] / (coef + 1) for coef in range(len(poly))]

    return [C] + integral
