#!/usr/bin/env python3
""" Module to build a ResNet-50 """
import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """ Resnet-50 with Keras """
    initializer = K.initializers.he_normal()
    X = K.Input(shape=(224, 224, 3))

    conv2d = K.layers.Conv2D(64, kernel_size=7,
                             strides=2, padding="same",
                             kernel_initializer=initializer)(X)
    batch_normalization = K.layers.BatchNormalization()(conv2d)
    activation = K.layers.Activation("relu")(batch_normalization)

    max_pool = K.layers.MaxPooling2D(pool_size=3, strides=2,
                                     padding="same")(activation)

    conv2d_1 = projection_block(max_pool, [64, 64, 256], 1)

    id_block = identity_block(conv2d_1, [64, 64, 256])
    id_block_1 = identity_block(id_block, [64, 64, 256])
    proj_block = projection_block(id_block_1, [128, 128, 512])

    id_block_2 = identity_block(proj_block, [128, 128, 512])
    id_block_3 = identity_block(id_block_2, [128, 128, 512])
    id_block_4 = identity_block(id_block_3, [128, 128, 512])
    proj_block_1 = projection_block(id_block_4, [256, 256, 1024])

    id_block_5 = identity_block(proj_block_1, [256, 256, 1024])
    id_block_6 = identity_block(id_block_5, [256, 256, 1024])
    id_block_7 = identity_block(id_block_6, [256, 256, 1024])
    id_block_8 = identity_block(id_block_7, [256, 256, 1024])
    id_block_9 = identity_block(id_block_8, [256, 256, 1024])
    proj_block_2 = projection_block(id_block_9, [512, 512, 2048])

    id_block_10 = identity_block(proj_block_2, [512, 512, 2048])
    id_block_11 = identity_block(id_block_10, [512, 512, 2048])
    avg_pool = K.layers.AveragePooling2D(pool_size=7, strides=1,
                                         padding="valid")(id_block_11)

    softmax = K.layers.Dense(units=1000, activation="softmax",
                             kernel_initializer=initializer)(avg_pool)
    return K.Model(inputs=X, outputs=softmax)
