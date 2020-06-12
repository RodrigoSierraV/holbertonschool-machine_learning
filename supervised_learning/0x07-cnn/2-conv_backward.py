#!/usr/bin/env python3
""" Module to compute back propagation in a convolutional layer with pooling"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """ Back prop for NN layer"""
    m, h_new, w_new, c_new = dZ.shape
    kh, kw, c_prev, c_new = W.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    sh, sw = stride
    if padding == 'same':
        ph = ((h_new - 1) * sh + kh - h_new) // 2
        pw = ((w_new - 1) * sw + kw - w_new) // 2
    else:
        ph, pw = 0, 0
    dW = np.zeros_like(W)
    for row in range(kh):
        for col in range(kw):
            dW[row, col, :, :] = np.sum(A_prev[:,
                                               row:row + h_new,
                                               col:col + w_new, :]
                                        * dZ[:, :, :, :],
                                        axis=(0, 1, 2)
                                        )
    db = np.sum(A_prev + b, axis=(1, 2, 3))
    dX = np.zeros_like(A_prev)
    pad_res = np.pad(dX, ((0, 0), (2 * ph, 2 * ph), (2 * pw, 2 * pw), (0, 0)),
                     mode='constant', constant_values=0)
    conv_h = (h_new - kh + 2 * ph) // sh + 1
    conv_w = (w_new - kw + 2 * pw) // sw + 1
    for row in range(conv_h):
        for col in range(conv_w):
            for n_k in range(c_prev):
                pad_res[:, row, col, n_k] = np.sum(dZ[:,
                                                      row*sh:row*sh + kh,
                                                      col*sw:col*sw + kw]
                                                   * W[:, :, :, n_k],
                                                   axis=(1, 2, 3))
    return pad_res, dW, db
