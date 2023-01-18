#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self, n=1000, initialize=True):
        """
        n: number of output classes
        initialize: whether to initialize weight and biases

        input size in the paper is 224 x 224 x 3

        first convolutional layer has kernel size 11 and stride 4
        so the output of first convolutional layer should be (224-11)/4 + 1 = 54
        but the paper claims the 2nd layer has size 55 x 55
        to solve this use padding of 2 or change input size to 227 x 227

        weights init -> 0 mean, 0.01 std, gaussian
        bias init -> 1 in 2nd, 4th, 5th layer
        bias init -> 0 in other layers
        """
        # init
        super().__init__()
        modules = []

        #### Convolutional layers
        #### 1st layer
        # input -> 224 x 224 x 3
        # (224 + 2*2(padding) - 11(kernel size)) / 4(stride) + 1 = 55
        # out -> 96 x 55 x 55
        modules.append(
                nn.Conv2d(
                    in_channels=3,
                    out_channels=96,
                    kernel_size=11,
                    stride=4,
                    padding=2,
                    )
                )
        if initialize:
            nn.init.normal_(modules[-1].weight, mean=0, std=0.01)
            nn.init.constant_(modules[-1].bias, 0)
        modules.append(nn.ReLU(inplace=True))
        # local response normalization
        modules.append(nn.LocalResponseNorm(k=2, size=5, alpha=1e-4, beta=0.75))
        # overlapping max pooling
        # input -> 96 x 55 x 55
        # (55 - 3(kernel size)) / 2(stride) + 1 = 27
        # out -> 96 x 27 x 27
        modules.append(nn.MaxPool2d(kernel_size=3, stride=2))
        ####

        #### 2nd layer
        # input -> 96 x 27 x 27
        # (27 + 2*2(padding) - 5(kernel size)) + 1 = 27
        # out -> 256 x 27 x 27
        modules.append(
                nn.Conv2d(
                    in_channels=96,
                    out_channels=256,
                    kernel_size=5,
                    padding=2,
                    )
                )
        if initialize:
            nn.init.normal_(modules[-1].weight, mean=0, std=0.01)
            nn.init.constant_(modules[-1].bias, 1)
        modules.append(nn.ReLU(inplace=True))
        # local response normalization
        modules.append(nn.LocalResponseNorm(k=2, size=5, alpha=1e-4, beta=0.75))
        # overlapping max pool
        # input -> 256 x 27 x 27
        # (27 - 3(kernel size)) / 2(stride) + 1 = 13
        # out -> 256 x 13 x 13
        modules.append(nn.MaxPool2d(kernel_size=3, stride=2))
        ####

        #### 3rd layer
        # input -> 256 x 13 x 13
        # 13 + 2*1(padding) - 3(kernel size) + 1 = 13
        # out -> 384 x 13 x 13
        modules.append(
                nn.Conv2d(
                    in_channels=256,
                    out_channels=384,
                    kernel_size=3,
                    padding=1,
                    )
                )
        if initialize:
            nn.init.normal_(modules[-1].weight, mean=0, std=0.01)
            nn.init.constant_(modules[-1].bias, 0)
        modules.append(nn.ReLU(inplace=True))
        ####

        #### 4th layer
        # input -> 384 x 13 x 13
        # out -> 384 x 13 x 13
        modules.append(
                nn.Conv2d(
                    in_channels=384,
                    out_channels=384,
                    kernel_size=3,
                    padding=1,
                    )
                )
        if initialize:
            nn.init.normal_(modules[-1].weight, mean=0, std=0.01)
            nn.init.constant_(modules[-1].bias, 1)
        modules.append(nn.ReLU(inplace=True))
        ####

        ####
        # 5th layer
        # input -> 384 x 13 x 13
        # out -> 256 x 13 x 13
        modules.append(
                nn.Conv2d(
                    in_channels=384,
                    out_channels=256,
                    kernel_size=3,
                    padding=1,
                    )
                )
        if initialize:
            nn.init.normal_(modules[-1].weight, mean=0, std=0.01)
            nn.init.constant_(modules[-1].bias, 1)
        modules.append(nn.ReLU(inplace=True))
        # overlapping max pool
        # input -> 256 x 13 x 13
        # (13 - 3(kernel size)) / 2(stride) + 1 = 6
        # out -> 256 x 6 x 6
        modules.append(nn.MaxPool2d(kernel_size=3, stride=2))
        ####
        self.conv_layers = nn.Sequential(*modules)

        #### fully-connected layers
        modules = []
        # dropout
        modules.append(nn.Dropout(p=0.5))
        # 1st linear layer
        modules.append(nn.Linear(256*6*6, 4096))
        if initialize:
            nn.init.normal_(modules[-1].weight, mean=0, std=0.01)
            nn.init.constant_(modules[-1].bias, 0)
        modules.append(nn.ReLU(inplace=True))
        # dropout
        modules.append(nn.Dropout(p=0.5))
        # 2nd linear layer
        modules.append(nn.Linear(4096, 4096))
        if initialize:
            nn.init.normal_(modules[-1].weight, mean=0, std=0.01)
            nn.init.constant_(modules[-1].bias, 0)
        modules.append(nn.ReLU(inplace=True))
        # last linear layer
        modules.append(nn.Linear(4096, n))
        if initialize:
            nn.init.normal_(modules[-1].weight, mean=0, std=0.01)
            nn.init.constant_(modules[-1].bias, 0)
        ####
        self.linear_layers = nn.Sequential(*modules)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, 256 * 6 * 6)
        x = self.linear_layers(x)
        return x
