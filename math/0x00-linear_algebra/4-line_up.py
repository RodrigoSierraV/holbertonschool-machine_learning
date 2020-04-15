#!/usr/bin/env python3
"""
Contains a method that adds two arrays
"""


def add_arrays(arr1, arr2):
    """
        Function that adds two arrays
        Return: A new array
    """
    if len(arr1) != len(arr2):
        return
    return [arr1[i] + arr2[i] for i in range(len(arr1))]
