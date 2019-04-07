#!/usr/bin/env python3.6
# -*- coding:utf-8 -*-
# data is from pyTorch framework
import gzip
import os
import pickle
import platform
from Act import ReLU
from Conv import Conv
from FC import FC
from Loss import SoftMaxCrossEntropyLoss
from Optimizer import SGD
from Pool import AvgPool
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from LayerBase import LayerBase
import numpy as np

np.random.seed(1)


class Net(LayerBase):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = FC(28 * 28, 256, 1e-3)
        self.relu1 = ReLU()
        self.fc2 = FC(256, 256, 1e-3)
        self.relu2 = ReLU()
        self.fc3 = FC(256, 10, 1e-3)
        # self.relu3 = ReLU()
        self.loss = SoftMaxCrossEntropyLoss()
        self.layers = [self.fc1, self.relu1, self.fc2, self.relu2, self.fc3, self.loss]

    def forward(self, x, y):
        if len(x.shape) != 2:
            x = np.reshape(x, (x.shape[0], -1))
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        lo = self.loss(x, y)
        return x, lo

    def backward(self, *param, **kwargs):
        pass


# load pickle based on python version 2 or 3
def load_pickle(f):
    version = platform.python_version_tuple()
    if version[0] == '2':
        return pickle.load(f)
    elif version[0] == '3':
        return pickle.load(f, encoding='latin1')
    raise ValueError("invalid python version: {}".format(version))


def load_mnist_datasets(path='./mnist.pkl.gz'):
    if not os.path.exists(path):
        raise Exception('Cannot find %s' % path)
    with gzip.open(path, 'rb') as f:
        train_set, val_set, test_set = load_pickle(f)
        return train_set, val_set, test_set


def to_categorical(y, num_classes=None):
    """从keras中复制而来
    Converts a class vector (integers) to binary class matrix.

    E.g. for use with categorical_crossentropy.

    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.

    # Returns
        A binary matrix representation of the input. The classes axis
        is placed last.
    """
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=np.float32)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


def get_accuracy(model, train_data, y_true):
    score, lo = model(train_data, y_true)
    acc = np.mean(np.argmax(score, axis=1) == y_true)
    return acc


def dnn_mnist():
    # load datasets
    path = 'mnist.pkl.gz'
    train_set, val_set, test_set = load_mnist_datasets(path)
    X_train, y_train = train_set

    X_val, y_val = val_set
    X_test, y_test = test_set

    # 转为稀疏分类
    y_train, y_val, y_test = to_categorical(y_train, 10), to_categorical(y_val, 10), to_categorical(y_test, 10)

    y_train = y_train.astype(int).argmax(axis=1)
    y_val = y_val.astype(int).argmax(axis=1)
    y_test = y_test.astype(int).argmax(axis=1)

    # bookeeping for best model based on validation set
    best_val_acc = -1
    mnist = Net()
    # Train
    batch_size = 32
    lr = 1e-1
    gamma = [50, 100, 150, 200]
    milestones = 0.5
    log_interval = 100
    epochs = 300
    optimizer = SGD(mnist, lr=lr, milestones=milestones, gamma=gamma)

    ind = np.arange(0, np.shape(X_train)[0], 1)
    import random
    random.seed(1)
    ind = ind.tolist()
    random.shuffle(ind)
    ind = np.array(ind)
    for epoch in range(epochs):
        num_train = X_train.shape[0]
        num_batch = num_train // batch_size
        for batch in range(num_batch):
            # get batch data
            # batch_mask = np.random.choice(num_train, batch_size)
            # X_batch = X_train[batch_mask]
            # y_batch = y_train[batch_mask]
            X_batch = X_train[ind[batch_size * batch:batch_size * (batch + 1)]]
            y_batch = y_train[ind[batch_size * batch:batch_size * (batch + 1)]]
            # 前向及反向
            output, loss = mnist(X_batch, y_batch)
            if batch % 200 == 0:
                print("Epoch %2d Iter %3d Loss %.5f" % (epoch, batch, loss))
            optimizer.step()
        train_acc = get_accuracy(mnist, X_train, y_train)
        val_acc = get_accuracy(mnist, X_val, y_val)

        if (best_val_acc < val_acc):
            best_val_acc = val_acc

        # store best model based n acc_val
        print('Epoch finish. ')
        print('Train acc %.3f' % train_acc)
        print('Val acc %.3f' % val_acc)
        print('-' * 30)
        print('')

    print('Train finished. Best acc %.3f' % best_val_acc)
    test_acc = get_accuracy(mnist, X_test, y_test)
    print('Test acc %.3f' % test_acc)


if __name__ == '__main__':
    dnn_mnist()
