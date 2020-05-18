#!/usr/bin/env python3
""" Module to create training operation """
import tensorflow as tf


def create_train_op(loss, alpha):
    """ Training operation for the network """
    return tf.train.GradientDescentOptimizer(alpha).minimize(loss)
