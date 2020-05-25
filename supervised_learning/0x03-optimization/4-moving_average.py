#!/usr/bin/env python3
""" Computes Moving Average """


def moving_average(data, beta):
    """ calculates the weighted moving average of a data set """
    moving_avg = []
    vt = 0
    for i in range(len(data)):
        vt = beta * vt + (1 - beta) * data[i]
        moving_avg.append(vt / (1 - beta ** (i + 1)))
    return moving_avg
