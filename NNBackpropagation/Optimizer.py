#!/usr/bin/env python3.6
# -*- coding:utf-8 -*-
from bisect import bisect_right
from functools import reduce

import FC
import Loss

import abc


class Optimizer(metaclass=abc.ABCMeta):
    def __init__(self, net, lr, milestones, gamma: list):
        self.net = net.layers[::-1]  # reverse order
        self.lr = lr
        self.milestones = milestones
        self.gamma = gamma
        self.epoch = -1

    def backward(self):
        curr_grad = 0
        for i in self.net:
            curr_grad = i.backward(curr_grad)

    @abc.abstractmethod
    def step(self, *param, **kwargs):
        pass

    def lrScheduler(self, *param, **kwargs):
        posi = bisect_right(self.milestones, self.epoch)
        if posi == 0:
            return self.lr
        else:
            return self.lr * reduce(lambda x, y: x * y, self.gamma[:posi])


class SGD(Optimizer):

    def __init__(self, net, lr, milestones, gamma: list):
        super(SGD, self).__init__(net, lr, milestones, gamma)

    def step(self):
        self.epoch += 1

        # gradient computing
        self.backward()
        # gradient backward
        for i in self.net:  # last layer is loss function, it need not update parameters
            for j in range(len(i.grad)):
                i.para[j] = i.para[j] - self.lr * i.grad[j]
