#!/usr/bin/env python3
"""
Save and load Models
"""
import tensorflow.keras as K


def save_weights(network, filename, save_format='h5'):
    """
    saves an entire model
    network: the model to save
    filename: is the path of the file that the model should be saved to
    save_format: is the format in which the weights should be saved
    return: None
    """
    network.save_weights(filepath=filename, save_format=save_format)


def load_weights(network, filename):
    """
    loads an entire model
    network: the model to which the weights should be loaded
    filename: the path of the file that the model should be loaded from
    return: None
    """
    network.load_weights(filename)
