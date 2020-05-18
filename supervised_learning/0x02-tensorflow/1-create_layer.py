#!/usr/bin/env python3
""" Module to create layers """
import tensorflow as tf


def create_layer(prev, n, activation):
    """ Creates a layer of a neural network"""
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    new_layer = tf.layers.Dense(n, activation, kernel_initializer=init,
                                name='layer')
    return new_layer(prev)
