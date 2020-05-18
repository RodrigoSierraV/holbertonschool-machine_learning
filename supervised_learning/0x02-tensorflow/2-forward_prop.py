#!/usr/bin/env python3
""" Module to compute Forward Propagation """
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """ Forward propagation """
    for i in range(len(layer_sizes)):
        if i == 0:
            prev_layer = x
        prev_layer = create_layer(prev_layer, layer_sizes[i], activations[i])
    return prev_layer
