#!/usr/bin/env python3
""" Module to represent Binomial distribution """


class Binomial():
    """ Class that models Binomial distribution """

    def __init__(self, data=None, n=1, p=0.5):
        " Instance constructor "

        if data is None:
            if n < 0:
                raise ValueError("n must be a positive value")
            if p < 0 or p > 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            mean = sum(data) / len(data)
            dif_add = sum([(d - mean) ** 2 for d in data])
            var = (dif_add / len(data))
            self.p = 1 - (var / mean)
            self.n = int(round(mean / self.p))
            self.p = mean / self.n

    def factorial(self, num):
        """Computes factorial of a number"""
        return num * self.factorial(num - 1) if num > 1 else 1

    def pmf(self, k):
        """ PMF """
        k = int(k)

        if k < 0:
            return 0

        factorial_n = self.factorial(self.n)
        factorial_k = self.factorial(k)
        factorial_n_k = self.factorial(self.n - k)
        prob = (self.p ** k) * (1 - self.p) ** (self.n - k)

        return factorial_n * prob / (factorial_k * factorial_n_k)
