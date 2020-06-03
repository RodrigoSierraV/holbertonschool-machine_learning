#!/usr/bin/env python3
""" Module to create a one-hot encoder """
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """  Label vector to one-hot """
    return K.utils.to_categorical(labels, classes)
