#!/usr/bin/env python3
""" Gradient descent with momentum algorithm """


def update_variables_momentum(alpha, beta1, var, grad, v):
    """ updates a variable using the GD with momentum algorithm """
    new_moment = beta1 * v + (1 - beta1) * grad
    updated_var = var - alpha * Vdv
    return updated_var, new_moment
