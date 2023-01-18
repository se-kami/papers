#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
from model import ResNet, Block, Bottleneck

if __name__ == '__main__':
    resnet18 = ResNet(Block, [2, 2, 2, 2])
    print(resnet18(torch.rand(20, 3, 224, 224)).shape)
