#!/usr/bin/env python3.6
# -*- coding:utf-8 -*-
import numpy as np
from LayerBase import LayerBase


class Softmax(LayerBase):
    def __init__(self):
        super(Softmax, self).__init__()
        self.y_grad = None

    def forward(self, x):
        # x: B*numInput
        expX = np.exp(x)
        sumExpX = np.sum(expX, axis=1, keepdims=True)
        y = expX / sumExpX  # B*numInput

        y_grad = -y[:, :, np.newaxis] * y[:, np.newaxis, :]
        y_grad1 = np.zeros_like(y_grad)
        for j, i in enumerate(y):
            y_grad1[j] = np.diag(i * (1 - i) + i * i)
        self.y_grad = y_grad + y_grad1  # B*I*I
        self.y_grad = np.mean(self.y_grad, axis=2)
        return y

    def backward(self, grad):
        '''
        :param grad:B*numInput
        '''

        return self.y_grad * grad


class ReLU(LayerBase):
    def __init__(self):
        super(ReLU, self).__init__()
        self.y_grad = None

    def forward(self, x):
        x = np.where(x <= 0, 0, x)
        self.y_grad = (x != 0).astype(float)
        return x

    def backward(self):
        return self.y_grad
