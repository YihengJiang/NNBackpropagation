#!/usr/bin/env python3.6
# -*- coding:utf-8 -*-
# pooling operation is learned from https://github.com/yizt/numpy_neural_network/blob/master/nn/layers.py
import numpy as np
from LayerBase import LayerBase

import os
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


class MaxPool(LayerBase):
    def __init__(self, kernel=(1, 1), stride=(2, 2), padding=(0, 0)):
        super(MaxPool, self).__init__()
        self.kernel = kernel
        self.padding = padding
        self.stride = stride

    def forward(self, x):
        """
        最大池化前向过程
        :param x: 卷积层矩阵,形状(N,C,H,W)，N为batch_size，C为通道数
        :param kernel: 池化大小(k1,k2)
        :param stride: 步长
        :param padding: 0填充
        :return:
        """
        N, C, H, W = x.shape
        # 零填充
        padding_x = np.lib.pad(x, ((0, 0), (0, 0), (self.padding[0], self.padding[0]),
                                   (self.padding[1], self.padding[1])), 'constant', constant_values=0)

        # 输出的高度和宽度
        out_h = (H + 2 * self.padding[0] - self.kernel[0]) // self.stride[0] + 1
        out_w = (W + 2 * self.padding[1] - self.kernel[1]) // self.stride[1] + 1

        pool_x = np.zeros((N, C, out_h, out_w))
        # used for back propagation
        self.padding_x = padding_x
        self.size = [N, C, out_h, out_w]

        for n in np.arange(self.size[0]):
            for c in np.arange(self.size[1]):
                for i in np.arange(self.size[2]):
                    for j in np.arange(self.size[3]):
                        pool_x[n, c, i, j] = np.max(padding_x[n, c,
                                                    self.stride[0] * i:self.stride[0] * i + self.kernel[0],
                                                    self.stride[1] * j:self.stride[1] * j + self.kernel[1]])

        return pool_x

    def backward(self, grad):
        """
        最大池化反向过程
        :param grad：损失函数关于最大池化输出的损失
        :param x: 卷积层矩阵,形状(N,C,H,W)，N为batch_size，C为通道数
        :param kernel: 池化大小(k1,k2)
        :param stride: 步长
        :param padding: 0填充
        :return:
        """
        x_grad = np.zeros_like(self.padding_x)
        for n in np.arange(self.size[0]):
            for c in np.arange(self.size[1]):
                for i in np.arange(self.size[2]):
                    for j in np.arange(self.size[3]):
                        # 找到最大值的那个元素坐标，将梯度传给这个坐标
                        flat_idx = np.argmax(self.padding_x[n, c,
                                             self.stride[0] * i:self.stride[0] * i + self.kernel[0],
                                             self.stride[1] * j:self.stride[1] * j + self.kernel[1]])
                        h_idx = self.stride[0] * i + flat_idx // self.kernel[1]
                        w_idx = self.stride[1] * j + flat_idx % self.kernel[1]
                        x_grad[n, c, h_idx, w_idx] = grad[n, c, i, j]

        # 返回时剔除零填充
        return _remove_padding(x_grad, self.padding)


class AvgPool(LayerBase):
    def __init__(self, kernel=(1, 1), stride=(2, 2), padding=(0, 0)):
        super(AvgPool, self).__init__()
        self.kernel = kernel
        self.padding = padding
        self.stride = stride

    def forward(self, x):

        N, C, H, W = x.shape
        # 零填充
        padding_x = np.lib.pad(x, ((0, 0), (0, 0), (self.padding[0], self.padding[0]),
                                   (self.padding[1], self.padding[1])), 'constant', constant_values=0)

        # 输出的高度和宽度
        out_h = (H + 2 * self.padding[0] - self.kernel[0]) // self.stride[0] + 1
        out_w = (W + 2 * self.padding[1] - self.kernel[1]) // self.stride[1] + 1

        pool_x = np.zeros((N, C, out_h, out_w))
        # used for back propagation
        self.padding_x = padding_x
        self.size = [N, C, out_h, out_w]

        for n in np.arange(self.size[0]):
            for c in np.arange(self.size[1]):
                for i in np.arange(self.size[2]):
                    for j in np.arange(self.size[3]):
                        pool_x[n, c, i, j] = np.mean(padding_x[n, c,
                                                     self.stride[0] * i:self.stride[0] * i + self.kernel[0],
                                                     self.stride[1] * j:self.stride[1] * j + self.kernel[1]])

        return pool_x

    def backward(self, grad):
        x_grad = np.zeros_like(self.padding_x)
        for n in np.arange(self.size[0]):
            for c in np.arange(self.size[1]):
                for i in np.arange(self.size[2]):
                    for j in np.arange(self.size[3]):
                        x_grad[n, c,
                        self.stride[0] * i:self.stride[0] * i + self.kernel[0],
                        self.stride[1] * j:self.stride[1] * j + self.kernel[1]] = \
                            grad[n, c, i, j] / (self.kernel[0] * self.kernel[1])
        return _remove_padding(x_grad, self.padding)


class GlobalMaxPool(LayerBase):
    def __init__(self):
        super(GlobalMaxPool, self).__init__()

    def forward(self, x):
        self.size = x.shape
        self.x = x
        return np.max(np.max(x, axis=-1), -1)

    def backward(self, grad):
        x_grad = np.zeros(self.size)
        for n in np.arange(self.size[0]):
            for c in np.arange(self.size[1]):
                # 找到最大值所在坐标，梯度传给这个坐标
                idx = np.argmax(self.x[n, c, :, :])
                h_idx = idx // self.size[3]
                w_idx = idx % self.size[3]
                x_grad[n, c, h_idx, w_idx] = grad[n, c]
        return x_grad


class GlobalAvgPool(LayerBase):
    def __init__(self):
        super(GlobalAvgPool, self).__init__()

    def forward(self, x):
        self.size = x.shape
        return np.mean(np.mean(x, axis=-1), -1)

    def backward(self, grad):
        x_grad = np.zeros(self.size)
        for n in np.arange(self.size[0]):
            for c in np.arange(self.size[1]):
                x_grad[n, c] = grad[n, c] / self.size[2] * self.size[3]
        return x_grad
