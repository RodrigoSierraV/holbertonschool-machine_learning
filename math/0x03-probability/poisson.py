#!/usr/bin/env python3

"""
This module represents Poisson distribution
"""


class Poisson:
    """
    class that models the Poisson distribution
    """

    def __init__(self, data=None, lambtha=1.):
        """Class constructor
        """
        if data is None:
            self.lambtha = float(lambtha)
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
        else:
            self.lambtha = float(sum(data) / len(data))
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

    def factorial(self, num):
        """Computes factorial of a number"""
        return num * self.factorial(num - 1) if num > 1 else 1

    def pmf(self, k):
        """Calculates Probability Mass Function PMF”"""
        e = 2.7182818285
        k = int(k)

        if k < 0:
            return 0

        exp = (e**(self.lambtha * (-1)))
        average = self.lambtha**k
        factorial = self.factorial(k)
        pmf = (exp * average) / factorial

        return pmf

    def cdf(self, k):
        """cumulative distribution function"""
        if k < 0:
            return 0
        k = int(k)

        cdf = 0
        for i in range(k + 1):
            cdf += (self.pmf(i))

        return cdf
