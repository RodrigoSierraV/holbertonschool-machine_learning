#!/usr/bin/env python3
""" Module to compute back propagation in a convolutional layer with pooling"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """ Back prop for NN layer"""
    m, h_new, w_new, c_new = dZ.shape
    kh, kw, c_prev, _ = W.shape
    _, h_prev, w_prev, _ = A_prev.shape
    sh, sw = stride
    if padding == 'same':
        ph = ((h_new - 1) * sh + kh - h_new) // 2
        pw = ((w_new - 1) * sw + kw - w_new) // 2
    else:
        ph, pw = 0, 0
    pad_res = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                     mode='constant', constant_values=0)
    dW = np.zeros(W.shape)
    db = np.sum(dZ, axis=(1, 2, 3), keepdims=True)
    dA = np.zeros(A_prev.shape)
    for img in range(m):
        for row in range(h_new):
            for col in range(w_new):
                for ch in range(c_new):
                    a = row * sh
                    b = row * sh + kh
                    c = col * sw
                    d = col * sw + kw
                    dA[img, a:b, c:d, :] += W[:, :, :, ch]\
                                            * dZ[img, row, col, ch]
                    dW[:, :, :, ch] += pad_res[img, a:b, c:d, :]\
                                       * dZ[img, row, col, ch]
    return dA, dW, db
