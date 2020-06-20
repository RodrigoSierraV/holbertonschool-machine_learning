#!/usr/bin/env python3
""" Module to build a transition layer"""
import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """
    X is the output from the previous layer
    nb_filters is an integer representing the number of filters in X
    compression is the compression factor for the transition layer
    All convolutions are preceded by Batch Normalization and
        rectified linear activation (ReLU)
    Returns: The output of the transition layer and the number of filters
        within the output, respectively
    """
    initializer = K.initializers.he_normal()
    F = int(nb_filters * compression)
    batch_norm = K.layers.BatchNormalization()(X)
    activation = K.layers.Activation("relu")(batch_norm)
    conv2d = K.layers.Conv2D(filters=F, padding="same",
                             kernel_size=1, strides=1,
                             kernel_initializer=initializer)(activation)
    avg_pool = K.layers.AveragePooling2D(pool_size=2, strides=2,
                                         padding="same")(conv2d)

    return avg_pool, F
