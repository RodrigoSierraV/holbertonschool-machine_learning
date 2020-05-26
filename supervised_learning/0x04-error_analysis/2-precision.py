#!/usr/bin/env python3
"""Precision of a Confusion matrix """
import numpy as np


def precision(confusion):
    """Calculates precision of confusion matrix
        precision = TP / (TP + FP)
    """
    prec = np.zeros(confusion.shape[0])
    for i in range(confusion.shape[0]):
        tp = confusion[i, i]
        fp = sum(confusion[:, i]) - tp
        prec[i] = tp / (tp + fp)
    return prec
