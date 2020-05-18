#!/usr/bin/env python3
""" Module to create placeholders """
import tensorflow as tf


def create_placeholders(nx, classes):
    """ Creates two placeholders """
    x = tf.placeholder(tf.float32, shape=(None, nx), name='x')
    y = tf.placeholder(tf.float32, shape=(None, classes), name='y')
    return x, y
