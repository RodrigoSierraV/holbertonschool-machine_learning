#!/usr/bin/env python3
""" Module to calculate Accuracy """
import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """ calculates the accuracy of predicted values """
    predicted = tf.argmax(y_pred, 1)
    true_values = tf.argmax(y, 1)
    equal = tf.equal(predicted, true_values)
    return tf.reduce_mean(tf.cast(equal, tf.float32))
