#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch import nn


class LeNet(nn.Module):
    def __init__(self, n_out=10):
        super().__init__()
        net = []

        # Layer 1
        # in -> 32x32
        # out -> 32 - kernel_size + 1 = 28x28
        net.append(nn.Conv2d(
            in_channels=1,
            out_channels=6,
            kernel_size=(5,5),
            stride=(1,1)))
        net.append(nn.ReLU(inplace=False))
        # pooling layer
        # in -> 28x28
        # out -> 14x14
        net.append(nn.AvgPool2d(kernel_size=(2,2), stride=(2,2)))

        # Layer 2
        # in -> 14x14
        # out -> 14 - kernel_size + 1 = 10
        net.append(nn.Conv2d(
            in_channels=6,
            out_channels=16,
            kernel_size=(5,5),
            stride=(1,1)))
        net.append(nn.ReLU(inplace=False))
        # pooling layer
        # in -> 10x10
        # out -> 5x5
        net.append(nn.AvgPool2d(kernel_size=(2,2), stride=(2,2)))

        # Layer 3
        # in -> 5x5
        # out -> 5 - kernel_size + 1 = 1
        net.append(nn.Conv2d(
            in_channels=16,
            out_channels=120,
            kernel_size=(5,5),
            stride=(1,1)))
        net.append(nn.ReLU(inplace=False))
        self.net = nn.Sequential(*net)

        # fc layers
        fc = []
        fc.append(nn.ReLU(inplace=False))
        fc.append(nn.Linear(120, 84))
        fc.append(nn.Linear(84, n_out))
        self.fc = nn.Sequential(*fc)

    def forward(self, x):
        # conv layers
        x = self.net(x)
        # flatten
        x = x.view(x.shape[0], -1)
        # fc layers
        x = self.fc(x)
        return x
