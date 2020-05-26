#!/usr/bin/env python3
"""Sensitivity of a Confusion matrix """
import numpy as np


def sensitivity(confusion):
    """Computes the sensitivity of confusion matrix
        sensitivity= TP / (TP + FN)
    """
    sens = np.zeros(confusion.shape[0])
    for i in range(confusion.shape[0]):
        tp = confusion[i, i]
        fn = sum(confusion[i, :]) - tp
        sens[i] = tp/(tp+fn)
    return sens
