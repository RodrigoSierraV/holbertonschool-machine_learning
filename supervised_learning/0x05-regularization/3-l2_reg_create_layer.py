#!/usr/bin/env python3
""" Module that create a Layer with L2 Regularization """
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """ Tensorflow layer with L2 regularization """
    initializer = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_AVG")
    regularizer = tf.contrib.layers.l2_regularizer(lambtha)
    layer = tf.layers.Dense(units=n,
                            activation=activation,
                            kernel_initializer=initializer,
                            kernel_regularizer=regularizer)
    return layer(prev)
