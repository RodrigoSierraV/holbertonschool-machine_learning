#!/usr/bin/env python3
""" Tensorflow LeNet-5 """
import tensorflow as tf


def lenet5(x, y):
    """ LeNet-5 modified arquitecture """
    init = tf.contrib.layers.variance_scaling_initializer()
    activation = tf.nn.relu
    conv_2d_1 = tf.layers.Conv2D(filters=6, kernel_size=5,
                                 padding="same", kernel_initializer=init,
                                 activation=activation)(x)

    pool_1 = tf.layers.MaxPooling2D(pool_size=[2, 2], strides=2)(conv_2d_1)

    conv_2d_2 = tf.layers.Conv2D(filters=16, kernel_size=5,
                                 padding="valid", kernel_initializer=init,
                                 activation=activation)(pool_1)

    pool_2 = tf.layers.MaxPooling2D(pool_size=[2, 2], strides=2)(conv_2d_2)

    flatten_1 = tf.layers.Flatten()(pool_2)

    full_layer_1 = tf.layers.Dense(units=120, kernel_initializer=init,
                                   activation=activation)(flatten_1)

    full_layer_2 = tf.layers.Dense(units=84, kernel_initializer=init,
                                   activation=activation,)(full_layer_1)

    last_layer = tf.layers.Dense(units=10,
                                 kernel_initializer=init)(full_layer_2)

    loss = tf.losses.softmax_cross_entropy(y, last_layer)
    minimize = tf.train.AdamOptimizer().minimize(loss)

    comp = tf.equal(tf.argmax(y, 1), tf.argmax(last_layer, 1))
    acc = tf.reduce_mean(tf.cast(comp, tf.float32))

    return tf.nn.softmax(last_layer), minimize, loss, acc
