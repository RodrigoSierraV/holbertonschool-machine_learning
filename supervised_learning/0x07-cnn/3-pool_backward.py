#!/usr/bin/env python3
""" Back propagation in a convolutional layer"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """ Back prop in a layer of CNN"""
    m, h_new, w_new, c_new = dA.shape
    kh, kw = kernel_shape
    sh, sw = stride
    dA_prev = np.zeros(A_prev.shape)
    for img in range(m):
        for row in range(h_new):
            for col in range(w_new):
                for ch in range(c_new):
                    a = row * sh
                    b = row * sh + kh
                    c = col * sw
                    d = col * sw + kw
                    if mode == "max":
                        a_slice = A_prev[img, a:b, c:d, ch]
                        mask = (a_slice == np.max(a_slice)).astype(int)
                        aux = dA[img, row, col, ch] * mask
                        dA_prev[img, a:b, c:d, ch] += aux
                    if mode == "avg":
                        average = dA[img, row, col, ch] / (kh * kw)
                        mask = np.ones(kernel_shape) * average
                        dA_prev[img, a:b, c:d, ch] += mask
    return dA_prev
