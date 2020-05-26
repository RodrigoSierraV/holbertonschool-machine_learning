#!/usr/bin/env python3
"""Specificity of a Confusion matrix """
import numpy as np


def specificity(confusion):
    """Computes specificity of confusion matrix
        specificity = TN / (TN + FP)
    """
    spec = np.zeros(confusion.shape[0])
    total = sum(confusion)
    print(total)
    for i in range(confusion.shape[0]):
        val = confusion[i, i]
        fp = sum(confusion[:, i]) - val
        tn = sum(total) - sum(confusion[:, i]) - sum(confusion[i, :]) + val
        spec[i] = tn / (tn + fp)
    return spec
