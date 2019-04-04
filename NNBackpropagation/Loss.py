#!/usr/bin/env python3.6
# -*- coding:utf-8 -*-
import numpy as np
from LayerBase import LayerBase


class MSELoss(LayerBase):

    def __init__(self):
        super(MSELoss, self).__init__()
        self.x_grad = None

    def forward(self, x, y):
        '''
        0.5 is for convenience of backward
        '''
        mse = 0.5 * np.sum((x - y) ** 2) / sum(np.shape(x))
        self.x_grad = x - y  # B*I
        return mse

    def backward(self, grad):
        '''
        for consistency with other layers, so tranfer a param:grad, it need not this param in fact
        '''
        return self.x_grad


class CrossEntropyLoss(LayerBase):
    '''
    this class dose not include softmax
    '''

    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.x_grad = None

    def forward(self, x, y):
        '''
        :param x:B*I
        :param y: not one hot vector
        :return:
        '''
        ce = -np.log(x[range(y.shape[0]), y])
        ce = np.mean(ce)
        self.x_grad = -1 / x

        return ce

    def backward(self):
        return self.x_grad


class SoftMaxCrossEntropyLoss(LayerBase):
    '''
    this class include softmax
    '''

    def __init__(self):
        super(SoftMaxCrossEntropyLoss, self).__init__()
        self.x_grad = None

    def forward(self, x, y):
        '''
        :param x:B*I
        :param y: not one hot vector
        :return:
        '''
        # softmax
        expX = np.exp(x)
        sumExpX = np.sum(expX, axis=1, keepdims=True)
        soft = expX / sumExpX  # B*numInput
        # ce
        ce = -np.log(soft[range(y.shape[0]), y])
        ce = np.mean(ce)
        #grad
        soft[range(y.shape[0]), y] = soft[range(y.shape[0]), y] - 1
        self.x_grad = soft

        return ce

    def backward(self):
        return self.x_grad
