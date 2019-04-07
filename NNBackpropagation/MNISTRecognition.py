#!/usr/bin/env python3.6
# -*- coding:utf-8 -*-
# data is from pyTorch framework

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
# np.random.seed(1994)
# torch.manual_seed(1994)


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



def train(epoch, train_loader, model, optimizer, log_interval):
    print("lr: ", optimizer.lr)
    for batch_idx, (data, target) in enumerate(train_loader):
        data = np.array(data.data)
        target = np.array(target.data)
        output, loss = model(data, target)
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss))


def test(test_loader, model):
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data = np.array(data.data)
        target = np.array(target.data)
        output, loss = model(data, target)
        test_loss += loss * len(target)
        acc, correct_curr = computeAcc(output, target)
        correct += correct_curr

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def computeAcc(x, y):
    pred_y = np.argmax(x, 1)
    right = float(sum(pred_y == y))
    acc = right / y.size
    return acc, right


def main():
    batch_size = 32
    lr = 0.1
    gamma = [50, 100, 150, 200]
    milestones = 0.5
    log_interval = 100
    epochs = 10

    # transform = transforms.Compose([transforms.ToTensor(),
    #                                 transforms.Normalize((0.1307,), (0.3081,))])
    transform = transforms.ToTensor()
    kwargs = {'num_workers': 20, 'pin_memory': True}
    train_dataset = datasets.MNIST('./mnist', train=True, transform=transform)
    test_dataset = datasets.MNIST('./mnist', train=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

    model = Net()
    optimizer = SGD(model, lr=lr, milestones=milestones, gamma=gamma)

    for epoch in range(epochs):
        train(epoch, train_loader, model, optimizer, log_interval)
        test(test_loader, model)


if __name__ == '__main__':
    main()
