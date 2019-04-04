#!/usr/bin/env python3.6
# -*- coding:utf-8 -*-

import numpy as np
import abc


class LayerBase(metaclass=abc.ABCMeta):

    def __init__(self, *param, **kwargs):
        self.grad = []


    @abc.abstractmethod
    def forward(self, *param, **kwargs):
        pass

    @abc.abstractmethod
    def backward(self, *param, **kwargs):
        pass
