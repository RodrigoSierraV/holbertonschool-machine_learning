#!/usr/bin/env python3
""" RMSProp optimizer """


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """ variable update with RMSProp algorithm """
    vdw = beta2 * s + (1 - beta2) * (grad ** 2)
    updated_var = var - alpha * (grad / (vdw ** (1/2) + epsilon))
    return updated_var, vdw
