#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from model import Transformer

def main(N, size_1, size_2):
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    vocab_size = N
    vocab_size_ = N
    x = torch.randint(N, size=(size_1, size_2), device=device)
    x_ = torch.randint(N, size=(size_1, size_2-1), device=device)
    model = Transformer(vocab_size, vocab_size_, device=device)
    y = model(x, x_)
    return y.shape


if __name__ == '__main__':
    N = 100
    size_1 = 30
    size_2 = 30
    print(main(N, size_1, size_2))
    print(f"Expected: {size_1} {size_2-1} {N}")
