#!/usr/bin/env python3
""" Module to create a neural network with keras input """
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """ Create a neural network """
    inputs = K.Input(shape=(nx,))
    x = K.layers.Dense(
        layers[0],
        input_shape=(nx,),
        activation=activations[0],
        kernel_regularizer=K.regularizers.l2(lambtha))(inputs)
    for layer, activation in zip(layers[1:], activations[1:]):
        x = K.layers.Dropout(1 - keep_prob)(x)
        x = K.layers.Dense(
            layer,
            activation=activation,
            kernel_regularizer=K.regularizers.l2(lambtha))(x)
    return K.models.Model(inputs=inputs, outputs=x)
