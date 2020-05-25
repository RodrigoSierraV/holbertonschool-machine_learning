#!/usr/bin/env python3
""" Module to compute standardization constants"""
import numpy as np


def normalization_constants(X):
    """Calculates standardization constants of a matrix X"""
    return np.mean(X, axis=0), np.std(X, axis=0)
