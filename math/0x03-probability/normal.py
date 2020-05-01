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
