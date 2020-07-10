#!/usr/bin/env python3
""" Create class Triplet Loss """
from tensorflow.keras.layers import Layer
import tensorflow.keras as K
import numpy as np


class TripletLoss(Layer):
    """ Class Triplet Loss definition and methods"""
    def __init__(self, alpha, **kwargs):
        """ Instance constructor
        alpha is the alpha value used to calculate the triplet loss
        """
        super(TripletLoss, self).__init__(**kwargs)
        self.alpha = alpha

    def triplet_loss(self, inputs):
        """ Compute Triplet Loss
        inputs is a list containing the anchor, positive and negative output
            tensors from the last layer of the model, respectively
        Returns: a tensor containing the triplet loss values
        """
        A, P, N = inputs
        res1 = K.backend.sum((A - P) ** 2, axis=1)
        res2 = K.backend.sum((A - N) ** 2, axis=1)
        return K.backend.maximum(res1 - res2, 0)

    def call(self, inputs):
        """
        inputs is a list containing the anchor, positive, and negative output
            tensors from the last layer of the model, respectively
        adds the triplet loss to the graph
        Returns: the triplet loss tensor
        """
        loss = self.triplet_loss(inputs)
        self.add_loss(loss)
        return loss
