#!/usr/bin/env python3
""" Module that create a Dense block"""
import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
    X is the output from the previous layer
    nb_filters is an integer representing the number of filters in X
    growth_rate is the growth rate for the dense block
    layers is the number of layers in the dense block
    Returns: The concatenated output of each layer within the Dense Block
        and the number of filters within the concatenated outputs, respectively
    """
    initialzer = K.initializers.he_normal(seed=None)
    for my_layer in range(layers):
        batch_norm = K.layers.BatchNormalization()(X)
        activation = K.layers.Activation('relu')(batch_norm)
        conv2d = K.layers.Conv2D(growth_rate * 4,
                                 kernel_size=1,
                                 padding='same',
                                 kernel_initializer=initialzer)(activation)
        batch_norm_1 = K.layers.BatchNormalization()(conv2d)
        activation_1 = K.layers.Activation('relu')(batch_norm_1)
        conv2d_1 = K.layers.Conv2D(
            growth_rate, kernel_size=3,
            padding='same',
            kernel_initializer=initialzer)(activation_1)
        X = K.layers.concatenate([X, conv2d_1])
        nb_filters = nb_filters + growth_rate
    return X, nb_filters
