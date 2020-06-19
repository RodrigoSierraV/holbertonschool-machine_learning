#!/usr/bin/env python3
""" Module to create an inception Block """
import tensorflow.keras as K


def inception_block(A_prev, filters):
    """
    A_prev is the output from the previous layer
    filters is a tuple or list containing F1, F3R, F3,F5R, F5, FPP:
        F1 is the number of filters in the 1x1 convolution
        F3R number of filters in 1x1 convolution before the 3x3 convolution
        F3 is the number of filters in the 3x3 convolution
        F5R number of filters in 1x1 convolution before the 5x5 convolution
        F5 is the number of filters in the 5x5 convolution
        FPP number of filters in the 1x1 convolution after the max pooling
    All convolutions inside inception block use rectified linear activation
    Returns: the concatenated output of the inception block
    """
    F1, F3R, F3, F5R, F5, FPP = filters
    initializer = K.initializers.he_normal(seed=None)

    f1 = K.layers.Conv2D(filters=F1, padding='same',
                         kernel_size=1, activation='relu',
                         kernel_initializer=initializer)(A_prev)

    f3r = K.layers.Conv2D(filters=F3R, padding='same',
                          kernel_size=1, activation='relu',
                          kernel_initializer=initializer)(A_prev)

    f3 = K.layers.Conv2D(filters=F3, padding='same',
                         kernel_size=3, activation='relu',
                         kernel_initializer=initializer)(f3r)

    f5r = K.layers.Conv2D(filters=F5R, padding='same',
                          kernel_size=1, activation='relu',
                          kernel_initializer=initializer)(A_prev)

    f5 = K.layers.Conv2D(filters=F5, padding='same',
                         kernel_size=5, activation='relu',
                         kernel_initializer=initializer)(f5r)

    max_pool = K.layers.MaxPool2D(pool_size=3, padding='same',
                                  strides=1)(A_prev)

    fpp = K.layers.Conv2D(filters=FPP, padding='same',
                          kernel_size=1, activation='relu',
                          kernel_initializer=initializer)(max_pool)

    return K.layers.concatenate([f1, f3, f5, fpp])
