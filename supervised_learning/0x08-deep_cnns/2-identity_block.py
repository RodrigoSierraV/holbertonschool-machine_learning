#!/usr/bin/env python3
""" Module to build an identity block """
import tensorflow.keras as K


def identity_block(A_prev, filters):
    """
    A_prev is the output from the previous layer
    filters is a tuple or list containing F11, F3, F12, respectively:
        F11 is the number of filters in the first 1x1 convolution
        F3 is the number of filters in the 3x3 convolution
        F12 is the number of filters in the second 1x1 convolution
    All convolutions inside the block are followed by batch normalization
        along the channels axis and a rectified linear activation (ReLU).
    All weights use he normal initialization
    Returns: the activated output of the identity block
    """
    initializer = K.initializers.he_normal()
    F11, F3, F12 = filters

    conv2d = K.layers.Conv2D(F11, kernel_size=1, padding="same",
                             kernel_initializer=initializer)(A_prev)
    batch_normalization = K.layers.BatchNormalization()(conv2d)
    activation = K.layers.Activation("relu")(batch_normalization)

    conv2d_1 = K.layers.Conv2D(F3, kernel_size=3, padding="same",
                               kernel_initializer=initializer)(activation)
    batch_normalization_1 = K.layers.BatchNormalization()(conv2d_1)
    activation_1 = K.layers.Activation("relu")(batch_normalization_1)

    conv2d_2 = K.layers.Conv2D(F12, kernel_size=1, padding="same",
                               kernel_initializer=initializer)(activation_1)
    batch_normalization_2 = K.layers.BatchNormalization()(conv2d_2)

    X = K.layers.Add()([batch_normalization_2, A_prev])

    return K.layers.Activation("relu")(X)
