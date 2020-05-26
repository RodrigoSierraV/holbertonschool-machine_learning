#!/usr/bin/env python3
"""Confusion matrix module"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """creates a confusion matrix"""
    conf_mat = np.zeros((labels.shape[1], labels.shape[1]))
    for i in range(len(labels)):
        conf_mat[labels[i].argmax()] += logits[i]
    return conf_mat
