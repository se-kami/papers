#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch import nn


class VGG(nn.Module):
    """
    M -> maxpool layer
    """
    if True:
        name_to_features = dict()
        name_to_features['A'] = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512]
        prev = name_to_features['A']
        name_to_features['B'] = prev[:1] + [64] + prev[1:3] + [128] + prev[3:]
        prev = name_to_features['B']
        name_to_features['C'] = prev[:8] + ['256-1'] + prev[8:11] + ['512-1'] + prev[11:16] + ['512-1']
        prev = name_to_features['C']
        name_to_features['D'] = prev[::]
        prev = name_to_features['D']
        prev[8] = 256
        prev[12] = 512
        prev[16] = 512
        name_to_features['E'] = prev[:9] + [256] + prev[9:13] + [512] + prev[13:] + [512]

    def __init__(self, in_channels=3, name='A', out_dim=10):
        super().__init__()
        self.in_channels = in_channels
        self.out_dim = out_dim
        features = VGG.name_to_features[name]

        def make_layers(features):
            layers = []
            in_channels = self.in_channels
            for f in features:
                if f == 'M':
                    layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]
                else:
                    if '-' in str(f):
                        kernel_size = 1
                        f = int(f.split('-')[0])
                    else:
                        kernel_size = 3

                    layers.append(nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=f,
                        kernel_size=(kernel_size, kernel_size),
                        padding=(1, 1),
                        stride=(1, 1),))
                    layers.append(nn.BatchNorm2d(f))
                    layers.append(nn.ReLU())
                    in_channels = f
            return layers

        layers = make_layers(features)
        layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]
        self.net = nn.Sequential(*layers)

        # fc layer
        fc = []
        fc.append(nn.Linear(512*7*7, 4096))
        fc.append(nn.ReLU())
        fc.append(nn.Dropout(p=0.5))
        fc.append(nn.Linear(4096, 4096))
        fc.append(nn.ReLU())
        fc.append(nn.Dropout(p=0.5))
        fc.append(nn.Linear(4096, self.out_dim))
        self.fc = nn.Sequential(*fc)


    def forward(self, x):
        x = self.net(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x
