#!/usr/bin/env python3
"""Module to create a one-hot decoder"""

import numpy as np


def one_hot_decode(one_hot):
    """Decodes one_hot"""
    try:
        return np.argmax(one_hot, axis=0)
    except Exception:
        return None
