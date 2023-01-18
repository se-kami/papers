#!/usr/bin/env python
# -*- coding: utf-8 -*-


import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            **kwargs,
            ):
        super().__init__()

        self.net = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                **kwargs)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.net(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class BasicBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            inner_channels,
            ):

        super().__init__()

        self.path_1 = ConvBlock(
                in_channels=in_channels,
                out_channels=inner_channels[0],
                kernel_size=1,
                )

        path_2 = []
        path_2.append(
                ConvBlock(
                    in_channels=in_channels,
                    out_channels=inner_channels[1],
                    kernel_size=1))
        path_2.append(
                ConvBlock(
                    in_channels=inner_channels[1],
                    out_channels=inner_channels[2],
                    kernel_size=3,
                    padding=1,))
        self.path_2 = nn.Sequential(*path_2)


        path_3 = []
        path_3.append(
                ConvBlock(
                    in_channels=in_channels,
                    out_channels=inner_channels[3],
                    kernel_size=1,))
        path_3.append(
                ConvBlock(
                    in_channels=inner_channels[3],
                    out_channels=inner_channels[4],
                    kernel_size=5,
                    padding=2,))
        self.path_3 = nn.Sequential(*path_3)

        path_4 = []
        path_4.append(nn.MaxPool2d(kernel_size=3, stride=1, padding=1))
        path_4.append(
                ConvBlock(
                    in_channels=in_channels,
                    out_channels=inner_channels[5],
                    kernel_size=1,))
        self.path_4 = nn.Sequential(*path_4)

    def forward(self, x):
        xs = []
        xs.append(self.path_1(x))
        xs.append(self.path_2(x))
        xs.append(self.path_3(x))
        xs.append(self.path_4(x))
        x = torch.cat(xs, dim=1)
        return x


class AuxBlock(nn.Module):
    def __init__(self, n_in, n_out, n_hidden=1024, p=0.7):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.net = ConvBlock(in_channels=n_in, out_channels=128, kernel_size=1)

        fc = []
        fc.append(nn.Linear(2048, n_hidden))
        fc.append(nn.ReLU())
        fc.append(nn.Dropout(0.7))
        fc.append(nn.Linear(n_hidden, n_out))
        self.fc = nn.Sequential(*fc)

    def forward(self, x):
        x = self.pool(x)
        x = self.net(x)
        # flatten
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x


class InceptionNet(nn.Module):
    def __init__(self, n_out=10, p=0.4):
        super().__init__()

        pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        net = []
        net.append(ConvBlock(
            in_channels=3,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            ))
        net.append(pool)
        net.append(ConvBlock(
            in_channels=64,
            out_channels=192,
            kernel_size=3,
            stride=1,
            padding=1,
            ))
        net.append(pool)

        arch = [[192, [64, 96, 128, 16, 32, 32]],
                [256, [128, 128, 192, 32, 96, 64]],
                'pool',
                [480, [192, 96, 208, 16, 48, 64]], ]

        for a in arch:
            if a == 'pool':
                net.append(pool)
            else:
                net.append(BasicBlock(*a))
        self.net_1 = nn.Sequential(*net)
        self.anet = AuxBlock(512, n_out)
        ############
        net = []
        arch = [[512, [160, 112, 224, 24, 64, 64]],
                [512, [128, 128, 256, 24, 64, 64]],
                [512, [112, 144, 288, 32, 64, 64]], ]
        for a in arch:
            if a == 'pool':
                net.append(pool)
            else:
                net.append(BasicBlock(*a))
        self.net_2 = nn.Sequential(*net)
        self.anet_ = AuxBlock(528, n_out)
        ############
        net = []
        arch = [[528, [256, 160, 320, 32, 128, 128]],
                'pool',
                [832, [256, 160, 320, 32, 128, 128]],
                [832, [384, 192, 384, 48, 128, 128]], ]
        for a in arch:
            if a == 'pool':
                net.append(pool)
            else:
                net.append(BasicBlock(*a))

        net.append(nn.AvgPool2d(kernel_size=7, stride=1))
        self.net_3 = nn.Sequential(*net)
        ############
        fc = []
        fc.append(nn.Dropout(p=p))
        fc.append(nn.Linear(1024, n_out))
        self.fc = nn.Sequential(*fc)


    def forward(self, x):
        ax, ax_ = None, None

        x = self.net_1(x)
        if self.training:
            ax = self.anet(x)

        x = self.net_2(x)
        if self.training:
            ax_ = self.anet_(x)

        x = self.net_3(x)
        # flatten
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x, ax, ax_
