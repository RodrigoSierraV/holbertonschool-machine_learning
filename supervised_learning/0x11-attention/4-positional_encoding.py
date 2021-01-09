#!/usr/bin/env python3
"""
Calculates the positional encoding for a transformer
"""
import numpy as np


def positional_encoding(max_seq_len, dm):
    """
    max_seq_len is an integer representing the maximum sequence length
    dm is the model depth
    Returns: a numpy.ndarray of shape (max_seq_len, dm) containing the
        positional encoding vectors
    """
    pos_embeddings = np.zeros((max_seq_len, dm))

    for position in range(max_seq_len):
        for i in range(0, dm, 2):
            div = np.exp(i * -np.log(10000.0) / dm)
            pos_embeddings[position, i] = (np.sin(position * div))
            pos_embeddings[position, i + 1] = (np.cos(position * div))
    return pos_embeddings
