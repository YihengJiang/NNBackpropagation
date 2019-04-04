#!/usr/bin/env python3.6
# -*- coding:utf-8 -*-
import numpy as np
from LayerBase import LayerBase


class FC(LayerBase):
    def __init__(self, numInput, numOutput):
        super(FC, self).__init__()
        self.w = np.random.rand(numOutput, numInput)
        self.b = np.random.rand(numOutput)
        self.w_grad = None
        self.b_grad = None
        self.grad = [self.w_grad, self.b_grad]
        self.para = [self.w, self.b]

    def forward(self, x):
        self.w_grad = x  # B*numInput,the gradient of w
        y = np.matmul(x, self.w.T).squeeze(1) + self.b
        return y

    def backward(self, grad):
        '''
        :param grad:B*numOutput
        '''
        self.w_grad = np.matmul(grad.T, self.w_grad) / grad.shape[0]  # mean on B
        self.b_grad = np.mean(grad, axis=0)  # mean on B
        in_grad = np.matmul(grad, self.w)  # B*I
        return in_grad
