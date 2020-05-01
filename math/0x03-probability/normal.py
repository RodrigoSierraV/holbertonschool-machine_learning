#!/usr/bin/env python3
"""
Normal Distribution
"""


class Normal:
    """
    class that models Normal Distribution
    """

    def __init__(self, data=None, mean=0., stddev=1.):
        """ Class constructor """
        self.mean = float(mean)
        self.stddev = float(stddev)
        if data is None:
            if stddev <= 0:
                raise ValueError('stddev must be a positive value')
        else:
            self.mean = (sum(data) / len(data))
            dif_add = sum([(d - self.mean) ** 2 for d in data])
            self.stddev = (dif_add / len(data)) ** (1 / 2)
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')

    def z_score(self, x):
        """ z-score of a given x-value """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """ x-value of a given z-score"""
        return (z * self.stddev) + self.mean

    def pdf(self, x):
        """probability density function
        """
        e = 2.7182818285
        pi = 3.1415926536
        variance = self.stddev ** 2
        exp = -((x - self.mean) ** 2) / (2 * variance)
        density = (2 * pi * variance) ** (1 / 2)
        return (e ** exp) / density

    def cdf(self, x):
        """Calculates the value of the CDF for a given x-value"""
        a = (x - self.mean) / (self.stddev * (2**0.5))
        erf = self.erf(a)
        return (1 + erf) / 2

    def factorial(self, num):
        """Computes factorial of a number"""
        return num * self.factorial(num - 1) if num > 1 else 1

    def erf(self, x):
        """ error function """
        pi = 3.1415926536
        serie = 0
        for i in range(5):
            j = 2 * i + 1
            density = self.factorial(i) * j
            if j in [3, 7]:
                serie += -(x ** (j)) / density
            elif j in [1, 5, 9]:
                serie += (x ** (j)) / density
        return serie * 2 / (pi ** (1 / 2))
