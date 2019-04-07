#!/usr/bin/env python3.6
# -*- coding:utf-8 -*-
# conv operation is learned from https://github.com/yizt/numpy_neural_network/blob/master/nn/layers.py

import numpy as np
from LayerBase import LayerBase


class Conv(LayerBase):
    '''
    do not consider dilation
    '''

    def __init__(self, inputC, outputC, kernel=(1, 1), stride=(2, 2), padding=(0, 0)):
        super(Conv, self).__init__()
        self.para = [np.random.rand(inputC, outputC, *kernel),# (C*D*k1*k2),C:input channel; D:output channel
                     np.random.rand(outputC)]# D
        self.padding = padding
        self.stride = stride

    def forward(self, x):
        self.x = x
        y = conv_forward(x.astype(np.float64), self.para[1].astype(np.float64),
                         self.para[1], self.padding, self.stride)
        return y

    def backward(self, grad):
        '''
        :param grad:N*D*H*W
        '''
        N, C, H, W = self.x.shape
        C, D, k1, k2 = self.para[0].shape

        padding_grad = _insert_zeros(grad, self.stride)

        # 卷积核高度和宽度翻转180度
        flip_K = np.flip(self.para[0], 2)
        flip_K = np.flip(flip_K, 3)
        swap_flip_K = np.swapaxes(flip_K, 0, 1)  # D*C*k1*k2
        # 增加高度和宽度0填充
        padding_grad = np.lib.pad(padding_grad, ((0, 0), (0, 0), (k1 - 1, k1 - 1),
                                                      (k2 - 1, k2 - 1)), 'constant', constant_values=0)
        y_grad = conv_forward(padding_grad.astype(np.float64), swap_flip_K.astype(np.float64),
                          np.zeros((C,), dtype=np.float64))

        # 求卷积和的梯度dK
        swap_x = np.swapaxes(self.x, 0, 1)  # 变为(C,N,H,W)与
        w_grad = conv_forward(swap_x.astype(np.float64), padding_grad.astype(np.float64),
                              np.zeros((D,), dtype=np.float64))

        # 偏置的梯度
        b_grad = np.sum(np.sum(np.sum(grad, axis=-1), axis=-1), axis=0)  # 在高度、宽度上相加；批量大小上相加

        # 把padding减掉
        y_grad = _remove_padding(y_grad, self.padding)  # dz[:, :, padding[0]:-padding[0], padding[1]:-padding[1]]

        w_grad = w_grad / N
        b_grad = b_grad / N
        self.grad = [w_grad, b_grad]
        return y_grad


def conv_forward(z, K, b, padding=(0, 0), strides=(1, 1)):
    """
    多通道卷积前向过程
    :param z: 卷积层矩阵,形状(N,C,H,W)，N为batch_size，C为通道数
    :param K: 卷积核,形状(C,D,k1,k2), C为输入通道数，D为输出通道数
    :param b: 偏置,形状(D,)
    :param padding: padding
    :param strides: 步长
    :return: 卷积结果
    """
    padding_z = np.lib.pad(z, ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])), 'constant',
                           constant_values=0)
    N, _, height, width = padding_z.shape
    C, D, k1, k2 = K.shape
    # 步长不为1时，步长必须刚好能够被整除
    assert (height - k1) % strides[0] == 0
    assert (width - k2) % strides[1] == 0
    conv_z = np.zeros((N, D, 1 + (height - k1) // strides[0], 1 + (width - k2) // strides[1]))
    for n in np.arange(N):
        for d in np.arange(D):
            for h in np.arange(height - k1 + 1)[::strides[0]]:
                for w in np.arange(width - k2 + 1)[::strides[1]]:
                    conv_z[n, d, h // strides[0], w // strides[1]] = np.sum(
                        padding_z[n, :, h:h + k1, w:w + k2] * K[:, d]) + b[d]
    return conv_z


def _insert_zeros(dz, strides):
    """
    想多维数组最后两位，每个行列之间增加指定的个数的零填充
    :param dz: (N,D,H,W),H,W为卷积输出层的高度和宽度
    :param strides: 步长
    :return:
    """
    _, _, H, W = dz.shape
    pz = dz
    if strides[0] > 1:
        for h in np.arange(H - 1, 0, -1):
            for o in np.arange(strides[0] - 1):
                pz = np.insert(pz, h, 0, axis=2)
    if strides[1] > 1:
        for w in np.arange(W - 1, 0, -1):
            for o in np.arange(strides[1] - 1):
                pz = np.insert(pz, w, 0, axis=3)
    return pz


def _remove_padding(x, padding):
    """
    移除padding
    :param x: (N,C,H,W)
    :param paddings: (p1,p2)
    :return:
    """
    if padding[0] > 0 and padding[1] > 0:
        return x[:, :, padding[0]:-padding[0], padding[1]:-padding[1]]
    elif padding[0] > 0:
        return x[:, :, padding[0]:-padding[0], :]
    elif padding[1] > 0:
        return x[:, :, :, padding[1]:-padding[1]]
    else:
        return x
