#!/usr/bin/env python3
""" Module to standardize a matrix"""
import numpy as np


def normalize(X, m, s):
    """Standardize a matrix X"""
    return (X - m) / s
