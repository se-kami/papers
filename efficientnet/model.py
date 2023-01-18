#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import nn
import math


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, groups=1):
        super().__init__()
        net = []
        net.append(nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups)
            )
        net.append(nn.BatchNorm2d(out_channels))
        net.append(nn.SiLU())
        self.net = nn.Sequential(*net)

    def forward(self, x):
        x = self.net(x)
        return x


class InvResBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            exp=4,
            sq=4,
            p=0.8, # for stochastic depth
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.exp = exp
        self.p = p
        self.sq = sq

        self.exp_channels = self.in_channels * self.exp
        self.inner_channels = int(self.in_channels / self.sq)

        # whether to use stochastic depth
        self.stochastic = (self.in_channels == self.out_channels) and (self.stride == 1)

        net = []
        if self.exp != 1:
            net.append(BasicBlock(
                in_channels=self.in_channels,
                out_channels=self.exp_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                ))
        net.append(BasicBlock(
            in_channels=self.exp_channels,
            out_channels=self.exp_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            groups=self.exp_channels,
            ))
        net.append(SqExc(
            in_channels=self.exp_channels,
            inner_channels=self.inner_channels,
            ))
        net.append(nn.Conv2d(
            in_channels=self.exp_channels,
            out_channels=self.out_channels,
            kernel_size=1,
            ))
        net.append(nn.BatchNorm2d(out_channels))
        self.net = nn.Sequential(*net)

    def forward(self, x):
        if self.stochastic:
            if self.training:
                return self.net(x) + x
            # stochastic tensor
            r = torch.rand(x.shape[0]).reshape(-1, 1, 1, 1)
            r = r.to(x.device)
            r = r < self.p
            r = torch.rand(x.shape[0], 1, 1, 1, device=x.device) < self.p
            r = r / self.p
            return r*self.net(x) + x
        return self.net(x)


class SqExc(nn.Module):
    def __init__(self, in_channels, inner_channels):
        super().__init__()
        net = []
        net.append(nn.AdaptiveAvgPool2d(1))
        net.append(nn.Conv2d(in_channels=in_channels,
                             out_channels=inner_channels,
                             kernel_size=1,
                             ))
        net.append(nn.SiLU())
        net.append(nn.Conv2d(
            in_channels=inner_channels,
            out_channels=in_channels,
            kernel_size=1,
            ))
        net.append(nn.Sigmoid())
        self.net = nn.Sequential(*net)

    def forward(self, x):
        f = self.net(x)
        x = f * x
        return x


class EfficientNet(nn.Module):
    """

    """
    # expansion, channels, repeat, stride, kernel
    cnn_params = [[1, 16, 1, 1, 3],
                  [6, 24, 2, 2, 3],
                  [6, 40, 2, 2, 5],
                  [6, 80, 3, 2, 3],
                  [6, 112, 3, 1, 5],
                  [6, 192, 4, 2, 5],
                  [6, 320, 1, 1, 3],]

    # phi, size, p
    net_params = {
            'B0': (0.0, 224, 0.2),
            'B1': (0.5, 240, 0.2),
            'B2': (1.0, 260, 0.3),
            'B3': (2.0, 300, 0.3),
            'B4': (3.0, 380, 0.4),
            'B5': (4.0, 456, 0.4),
            'B6': (5.0, 528, 0.5),
            'B7': (6.0, 600, 0.5),
            }

    def __init__(self, name, n_out=10, alpha=1.2, beta=1.1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.phi, self.size, self.p = EfficientNet.net_params[name]
        w, d = self.alpha ** self.phi, self.beta ** self.phi

        net = []

        in_channels = 3
        out_channels = int(32 * w)
        net.append(BasicBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            ))
        in_channels = out_channels

        for param in EfficientNet.cnn_params:
            exp, channels, repeat, stride, kernel_size = param
            padding = int((kernel_size-1)/2)
            out_channels = 4*math.ceil(int(channels*w) / 4)
            n_layers = math.ceil(d * repeat)

            for layer in range(n_layers):
                if layer == 0:
                    stride = stride
                else:
                    stride = 1
                net.append(
                    InvResBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        exp=exp,
                        stride=stride,
                        kernel_size=kernel_size,
                        padding=padding,
                        p=self.p
                    )
                )
                in_channels = out_channels

        n_last = math.ceil(1280 * w)
        net.append(
            BasicBlock(
                in_channels=in_channels,
                out_channels=n_last,
                kernel_size=1,
                stride=1,
                padding=0)
        )
        net.append(nn.AdaptiveAvgPool2d(1))
        self.net = nn.Sequential(*net)

        # fc
        fc = []
        fc.append(nn.Dropout(self.p))
        fc.append(nn.Linear(n_last, n_out))
        self.fc = nn.Sequential(*fc)

    def forward(self, x):
        x = self.net(x)
        # flatten
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x
