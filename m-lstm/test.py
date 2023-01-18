#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from model import DoubleCellModel


if __name__ == '__main__':
    model = DoubleCellModel(128, 128, 5, 50, 0.5)
    optimizer = torch.optim.Adam(model.parameters())
    r = torch.randint(50, (9, 10))
    out, hidden = model(r)
    print(out.shape, hidden.shape)
