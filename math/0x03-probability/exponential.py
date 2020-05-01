#!/usr/bin/env python3
"""
Exponential distribution
"""


class Exponential:
    """
    class that models exponential distribution
    """

    def __init__(self, data=None, lambtha=1.):
        """initialize class
        """
        if data is None:
            self.lambtha = float(lambtha)
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = 1 / (float(sum(data) / len(data)))

    def pdf(self, x):
        """ PDF for a given time period """
        e = 2.7182818285
        if x < 0:
            return 0
        value = e ** ((-self.lambtha)*x)
        return self.lambtha * value
