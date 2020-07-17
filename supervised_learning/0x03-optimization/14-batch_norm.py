#!/usr/bin/env python3
""" Module for Batch Normalization with tf"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    prev is the activated output of the previous layer
    n is the number of nodes in the layer to be created
    activation is the activation function that should be used on the output
        of the layer
    Returns: a tensor of the activated output for the layer
    """
    weights = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_AVG")
    layer = tf.layers.Dense(n, kernel_initializer=weights)
    Z = layer(prev)
    beta = tf.Variable(tf.zeros([n]))
    gamma = tf.Variable(tf.ones([n]))
    mean, variance = tf.nn.moments(Z, axes=0)
    batch_norm = tf.nn.batch_normalization(Z, mean, variance,
                                           offset=beta, scale=gamma,
                                           variance_epsilon=1e-8)
    return activation(batch_norm)
