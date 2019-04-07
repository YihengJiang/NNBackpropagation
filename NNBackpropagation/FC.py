#!/usr/bin/env python3.6
# -*- coding:utf-8 -*-
import numpy as np
from LayerBase import LayerBase


class FC(LayerBase):
    def __init__(self, numInput, numOutput, initV=None):
        super(FC, self).__init__()
        if initV != None:
            self.para = [initV * np.random.randn(numInput,numOutput), np.zeros(numOutput)]
        else:
            self.para = [np.random.randn(numInput,numOutput), np.zeros(numOutput)]

    def forward(self, x):
        self.w_grad = x  # B*numInput,the gradient of w
        y = np.matmul(x, self.para[0]) + self.para[1]
        return y

    def backward(self, grad):
        '''
        :param grad:B*numOutput
        '''
        self.w_grad = np.matmul(grad.T, self.w_grad) / grad.shape[0]  # mean on B
        self.b_grad = np.mean(grad, axis=0)  # mean on B
        in_grad = np.matmul(grad, self.para[0].T)  # B*I
        self.grad = [self.w_grad.T, self.b_grad]
        return in_grad
