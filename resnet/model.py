#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch.nn as nn


class Block(nn.Module):
    exp = 1
    def __init__(self, in_channels, out_channels, stride, short=None):
        super().__init__()
        self.short = short

        net = []
        net.append(nn.Conv2d(in_channels=in_channels,
                             out_channels=out_channels,
                             kernel_size=3,
                             stride=stride,
                             padding=1))

        net.append(nn.BatchNorm2d(out_channels))
        net.append(nn.ReLU())
        net.append(nn.Conv2d(in_channels=out_channels,
                             out_channels=out_channels,
                             kernel_size=3,
                             stride=1,
                             padding=1))
        net.append(nn.BatchNorm2d(out_channels))
        self.net = nn.Sequential(*net)

        # last activation
        self.act = nn.ReLU()

    def forward(self, x):
        if self.short is not None:
            short = self.short(x)
        else:
            short = 0
        x = self.net(x)
        x = x + short
        x = self.act(x)
        return x


class Bottleneck(nn.Module):
    exp = 4
    def __init__(self, in_channels, mid_channels, stride, short=None):
        super().__init__()
        self.short = short
        out_channels = self.exp * mid_channels

        net = []

        net.append(nn.Conv2d(in_channels=in_channels,
                             out_channels=mid_channels,
                             kernel_size=1,
                             stride=1,))

        net.append(nn.BatchNorm2d(mid_channels))
        net.append(nn.ReLU())
        net.append(nn.Conv2d(in_channels=mid_channels,
                             out_channels=mid_channels,
                             kernel_size=3,
                             stride=stride,
                             padding=1))

        net.append(nn.BatchNorm2d(mid_channels))
        net.append(nn.ReLU())
        net.append(nn.Conv2d(in_channels=mid_channels,
                             out_channels=out_channels,
                             kernel_size=1,
                             stride=1,))
        net.append(nn.BatchNorm2d(out_channels))
        self.net = nn.Sequential(*net)

        self.act = nn.ReLU()

    def forward(self, x):
        if self.short is not None:
            short = self.short(x)
        else:
            short = 0
        x = self.net(x)
        x = x + short
        x = self.act(x)
        return x


class ResNet(nn.Module):
    def __init__(self, block, layers, in_chn=3, n_out=10, kernel=7):
        super().__init__()
        self.mid_chn = [64, 128, 256, 512]
        self.in_channels = self.mid_chn[0]
        self.strides = [1, 2, 2, 2]
        self.layers = layers
        net = []
        net.append(nn.Conv2d(in_chn, self.mid_chn[0], kernel_size=kernel, stride=2, padding=3))
        net.append(nn.BatchNorm2d(self.mid_chn[0]))
        net.append(nn.ReLU())
        net.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        layers = []
        for layer, mid_chn, stride in zip(self.layers, self.mid_chn, self.strides):
            layers.append(self.make_layer(block, layer, mid_chn, stride))

        net += layers
        self.net = nn.Sequential(*net)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.exp, n_out)

    def forward(self, x):
        x = self.net(x)
        x = self.pool(x)
        # flatten
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

    def make_layer(self, block, res_blocks, mid_chn, stride):
        short = None
        if stride != 1 or self.in_channels != mid_chn * block.exp:
            short = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    mid_chn * block.exp,
                    kernel_size=1,
                    stride=stride,
                ),
                nn.BatchNorm2d(mid_chn * block.exp),
            )

        layers = []
        layers.append(block(self.in_channels, mid_chn, stride, short=short))
        self.in_channels = mid_chn * block.exp

        for i in range(res_blocks - 1):
            layers.append(block(self.in_channels, mid_chn, stride))
        layers =  nn.Sequential(*layers)
        return layers


def resnet18(n_out=10):
    r = ResNet(Block, [2, 2, 2, 2], n_out=n_out)
    return r
