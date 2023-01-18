#!/usr/bin/env python
# -*- coding: utf-8 -*-


import torch
from torch import nn


class M_LSTM_Cell(nn.Module):
    def __init__(self, in_size, hidden_size, steps):
        super().__init__()
        self.steps = steps
        self.cell = nn.LSTMCell(in_size, hidden_size)

        net = []
        a, b = in_size, hidden_size
        for i in range(self.steps):
            net.append(nn.Linear(b, a))
            a, b = b, a
        self.net = nn.ModuleList(net)

    def mogrify(self, x, y):
        for i in range(self.steps):
            if i % 2 == 0:
                temp = torch.sigmoid(self.net[i](y))
                x = 2 * temp * x
            else:
                temp = torch.sigmoid(self.net[i](x))
                y = 2 * temp * y
        return x, y

    def forward(self, x, y):
        a, b = y
        x, a = self.mogrify(x, a)
        a, b = self.cell(x, (a, b))
        return a, b


class DoubleCellModel(nn.Module):
    def __init__(self,
                 in_size,
                 hidden_size,
                 steps,
                 vocab,
                 dropout,):

        super().__init__()
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.steps = steps
        self.vocab = vocab
        self.dropout = dropout

        self.emb = nn.Embedding(self.vocab, self.in_size)
        self.cell = M_LSTM_Cell(self.in_size, self.hidden_size, self.steps)
        self.cell_ = M_LSTM_Cell(self.hidden_size, self.hidden_size, self.steps)

        self.fc = nn.Linear(self.hidden_size, self.vocab)

        self.drop = nn.Dropout(self.dropout)

    def forward(self, x, L=10):

        size = x.shape[0]
        x = self.emb(x)
        a = torch.zeros(size, self.hidden_size)
        a_ = torch.zeros(size, self.hidden_size)
        b = torch.zeros(size, self.hidden_size)
        b_ = torch.zeros(size, self.hidden_size)

        out, hidden = [], []
        for i in range(L):
            x_emb = self.drop(x[:, i])
            a, b = self.cell(x_emb, (a, b))
            a_, b_ = self.cell_(a, (a_, b_))
            y = self.drop(b_)
            y = self.fc(y)
            y = y.unsqueeze(1)
            out.append(y)
            hidden.append(a_.unsqueeze(1))

        out = torch.cat(out, 1)
        hidden = torch.cat(hidden, 1)

        return out, hidden
