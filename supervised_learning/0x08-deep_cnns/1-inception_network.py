#!/usr/bin/env python3
""" Module to build an inception network """
import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """ Build an inception network with Keras"""
    X = K.Input(shape=(224, 224, 3))
    initializer = K.initializers.he_normal(seed=None)
    conv_1 = K.layers.Conv2D(filters=64, kernel_size=7,
                             padding='same', strides=2,
                             kernel_initializer=initializer,
                             activation='relu')(X)
    max_pool_1 = K.layers.MaxPooling2D(pool_size=3, strides=2,
                                       padding='same')(conv_1)

    conv_2 = K.layers.Conv2D(filters=64, padding='same',
                             kernel_size=1, activation='relu',
                             kernel_initializer=initializer)(max_pool_1)
    conv2_1 = K.layers.Conv2D(filters=192, padding='same',
                              kernel_size=3, activation='relu',
                              kernel_initializer=initializer)(conv_2)
    max_pool_2 = K.layers.MaxPooling2D(pool_size=3, strides=2,
                                       padding='same')(conv2_1)

    incep_3a = inception_block(max_pool_2, [64, 96, 128, 16, 32, 32])
    incep_3b = inception_block(incep_3a, [128, 128, 192, 32, 96, 64])
    max_pool_3 = K.layers.MaxPooling2D(pool_size=3, strides=2,
                                       padding='same')(incep_3b)

    incep_4a = inception_block(max_pool_3, [192, 96, 208, 16, 48, 64])
    incep_4b = inception_block(incep_4a, [160, 112, 224, 24, 64, 64])
    incep_4c = inception_block(incep_4b, [128, 128, 256, 24, 64, 64])
    incep_4d = inception_block(incep_4c, [112, 144, 288, 32, 64, 64])
    incep_4e = inception_block(incep_4d, [256, 160, 320, 32, 128, 128])
    max_pool_4 = K.layers.MaxPooling2D(pool_size=3, strides=2,
                                       padding='same')(incep_4e)

    incep_5a = inception_block(max_pool_4, [256, 160, 320, 32, 128, 128])
    incep_5b = inception_block(incep_5a, [384, 192, 384, 48, 128, 128])
    avg_pool = K.layers.AveragePooling2D(pool_size=7, strides=None)(incep_5b)

    drop_out = K.layers.Dropout(0.4)(avg_pool)
    dense = K.layers.Dense(units=1000, activation='softmax',
                           kernel_initializer=initializer)(drop_out)
    return K.models.Model(inputs=X, outputs=dense)
