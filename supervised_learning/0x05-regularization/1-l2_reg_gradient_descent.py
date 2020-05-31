#!/usr/bin/env python3
""" Gradient Descent with L2 Regularization """
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """ Back-prop with gradient descent and L2 regularization """
    w = weights.copy()
    m = Y.shape[1]
    for layer in range(L, 1, -1):
        A = 'A{}'.format(layer)
        A_prev = 'A{}'.format(layer - 1)
        w_lay = 'W{}'.format(layer)
        b = 'b{}'.format(layer)
        if layer == L:
            dz = cache[A] - Y
            dw = (np.matmul(cache[A_prev], dz.T) / m).T
        else:
            w_back = 'W{}'.format(layer + 1)
            d1 = np.matmul(w[w_back].T, dz_prev)
            derive_tanh = 1 - cache[A] ** 2
            dz = d1 * derive_tanh
            dw = np.matmul(dz, cache[A_prev].T) / m
        dz_prev = dz
        dw_reg = dw + (lambtha * w[w_lay] / m)
        db = np.sum(dz, axis=1, keepdims=True)

        weights[w_lay] = w[w_lay] - alpha * dw_reg
        weights[b] = w[b] - alpha * db
