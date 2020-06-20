#!/usr/bin/env python3
""" Module to create a DenseNet-121 """
import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """
    growth_rate is the growth rate
    compression is the compression factor
    Data will have shape (224, 224, 3)
    All convolutions are preceded by Batch Normalization and
        rectified linear activation (ReLU)
    All weights use he normal initialization
    Returns: the keras model
    """
    initializer = K.initializers.he_normal(seed=None)
    X = K.Input(shape=(224, 224, 3))
    layers = [12, 24, 16]
    batch_norm = K.layers.BatchNormalization()(X)
    activation = K.layers.Activation('relu')(batch_norm)
    conv2d = K.layers.Conv2D(filters=2 * growth_rate, padding='same',
                             kernel_size=7, strides=2,
                             kernel_initializer=initializer)(activation)

    max_pool = K.layers.MaxPool2D(pool_size=3, padding='same',
                                  strides=2, )(conv2d)
    y, nb_filters = dense_block(max_pool, 2 * growth_rate, growth_rate, 6)

    for layer in layers:
        tran, nb_filters = transition_layer(y, nb_filters, compression)
        y, nb_filters = dense_block(tran, nb_filters, growth_rate, layer)

    avg_pool = K.layers.AveragePooling2D(pool_size=7, padding='same',
                                         strides=1)(y)
    dense = K.layers.Dense(1000, activation='softmax',
                           kernel_regularizer=K.regularizers.l2())(avg_pool)
    return K.models.Model(inputs=X, outputs=dense)
