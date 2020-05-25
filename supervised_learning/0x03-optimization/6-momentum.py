#!/usr/bin/env python3
"""Module for Momentum Optimizer"""
import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """tensorflow momentum optimization algorithm"""

    return tf.train.MomentumOptimizer(
        alpha,
        beta1,
    ).minimize(loss)
