#!/usr/bin/env python3
""" Module to create a neural network with keras """
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """ Create a neural network """
    model = K.Sequential()
    model.add(
        K.layers.Dense(
            layers[0],
            input_shape=(nx,),
            activation=activations[0],
            kernel_regularizer=K.regularizers.l2(lambtha)
        )
    )
    for layer, activation in zip(layers[1:], activations[1:]):
        model.add(K.layers.Dropout(1 - keep_prob))
        model.add(
            K.layers.Dense(
                layer,
                activation=activation,
                kernel_regularizer=K.regularizers.l2(lambtha)
            )
        )
    return model
