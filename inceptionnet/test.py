#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from model import InceptionNet


if __name__ == "__main__":
    batch = 12
    n_out = 15
    model = InceptionNet(n_out=n_out)
    r = torch.randn(batch, 3, 224, 224)
    print(model(r)[0].shape)
    print(f"Expected: {batch} x {n_out}")
