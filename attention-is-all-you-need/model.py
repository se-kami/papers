#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import nn


class SelfAttention(nn.Module):
    def __init__(self, size, n_head, minf=-1e20):
        super().__init__()

        self.size = size
        self.n_head = n_head
        self.dim = self.size // self.n_head
        self.minf = minf  # minus infinity

        self.val = nn.Linear(self.size, self.size, bias=False)
        self.key = nn.Linear(self.size, self.size, bias=False)
        self.q = nn.Linear(self.size, self.size, bias=False)

        self.fc = nn.Linear(self.dim * self.n_head, self.size)


    def forward(self, vals, keys, qs, mask):
        size = qs.shape[0]

        vals = self.val(vals)
        keys = self.key(keys)
        qs = self.q(qs)

        # size, vals.shape[1], n_head, dim
        vals = vals.reshape(size, vals.shape[1], self.n_head, -1)  # last dim should be equal to self.dim
        # size, keys.shape[1], n_head, dim
        keys = keys.reshape(size, keys.shape[1], self.n_head, -1)  # last dim should be equal to self.dim
        # size, qs.shape[1], n_head, dim
        qs = qs.reshape(size, qs.shape[1], self.n_head, -1)  # last dim should be equal to self.dim

        # in -> (size x qs.shape[1] x n_head x dim) X (size x keys.shape[1] x n_head x dim)
        # out -> size x n_head x qs.shape[1] x keys.shape[1]
        x = torch.einsum("aibc,ajbc->abij", [qs, keys])

        # mask
        x = x.masked_fill(mask == 0, self.minf) if mask is not None else x

        # attention
        attn = torch.softmax(x / (self.size)**0.5, dim=3)

        # in -> (size x n_head x qs.shape[1] x keys.shape[1]) X (size x vals.shape[1] x n_head x dim)
        # out -> size x qs.shape[1] x n_head x dim
        x = torch.einsum("abic,acbd->aibd", [attn, vals])
        x = x.reshape(size, -1, self.size)  # n_head x dim = self.size

        # fc layer
        x = self.fc(x)

        return x


class TransformerBlock(nn.Module):
    def __init__(self, size, n_head, dropout=0, exp=4):
        super().__init__()
        self.size = size
        self.n_head = n_head
        self.dropout = dropout
        self.exp = exp

        self.attn = SelfAttention(self.size, self.n_head)

        fc = []
        fc.append(nn.Linear(self.size, self.size*self.exp))
        fc.append(nn.ReLU())
        fc.append(nn.Linear(self.size*self.exp, self.size))
        fc = nn.Sequential(*fc)
        self.fc = fc

        self.dropout_layer = nn.Dropout(self.dropout)
        self.norm = nn.LayerNorm(self.size)

    def forward(self, vals, keys, qs, mask):
        attn = self.attn(vals, keys, qs, mask)
        x = self.norm(attn + qs)
        x = self.dropout_layer(x)
        x = self.norm(x + self.fc(x))
        x = self.dropout_layer(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, size, n_head, dropout=0, exp=4):
        super().__init__()
        self.size = size
        self.n_head = n_head
        self.dropout = dropout
        self.exp = exp

        self.attn = SelfAttention(self.size, self.n_head)
        self.transform = TransformerBlock(self.size, self.n_head,
                                          dropout=self.dropout, exp=self.exp)

        self.dropout_layer = nn.Dropout(self.dropout)
        self.norm = nn.LayerNorm(self.size)

    def forward(self, x, vals, keys, mask, mask_):
        attn = self.attn(x, x, x, mask_)
        x = attn + x
        x = self.norm(x)
        x = self.dropout_layer(x)
        x = self.transform(vals, keys, x, mask)
        return x


class Encoder(nn.Module):
    def __init__(self, vocab_size, pos_size, size, n_head, n_layer,
                 dropout=0, exp=4, device=None):
        super().__init__()

        self.vocab_size = vocab_size
        self.pos_size = pos_size
        self.size = size
        self.n_head = n_head
        self.n_layer = n_layer
        self.dropout = dropout
        self.exp = exp
        self.device = device

        self.word_e = nn.Embedding(self.vocab_size, size)
        self.pos_e = nn.Embedding(self.pos_size, size)

        layers = []
        for i in range(n_layer):
            layers.append(
                    TransformerBlock(size=self.size, n_head=self.n_head,
                                     dropout=self.dropout, exp=self.exp)
                    )
        self.layers = nn.ModuleList(layers)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x, mask):
        # make pos tensor
        pos = torch.arange(0, x.shape[1])
        pos = pos.expand(x.shape[0], x.shape[1])
        if self.device is not None:
            pos = pos.to(self.device)
        # sum embeddings
        x = self.word_e(x) + self.pos_e(pos)
        x = self.dropout_layer(x)
        # run through transformer blocks
        for layer in self.layers:
            x = layer(x, x, x, mask)

        return x


class Decoder(nn.Module):
    def __init__(self, vocab_size, pos_size, size, n_head, n_layer,
                 dropout=0, exp=4, device=None):
        super().__init__()

        self.vocab_size = vocab_size
        self.pos_size = pos_size
        self.size = size
        self.n_head = n_head
        self.n_layer = n_layer
        self.dropout = dropout
        self.exp = exp
        self.device = device

        self.word_e = nn.Embedding(self.vocab_size, self.size)
        self.pos_e = nn.Embedding(self.pos_size, self.size)

        layers = []
        for i in range(n_layer):
            layers.append(
                    DecoderBlock(size=self.size, n_head=self.n_head,
                                 dropout=self.dropout, exp=self.exp,)
                    )
        self.layers = nn.ModuleList(layers)
        self.dropout_layer = nn.Dropout(dropout)
        self.fc = nn.Linear(self.size, self.vocab_size)

    def forward(self, x, x_enc, mask, mask_):
        # make pos tensor
        pos = torch.arange(0, x.shape[1])
        pos = pos.expand(x.shape[0], x.shape[1])
        if self.device is not None:
            pos = pos.to(self.device)
        # sum embeddings
        x = self.word_e(x) + self.pos_e(pos)
        x = self.dropout_layer(x)
        for layer in self.layers:
            x = layer(x, x_enc, x_enc, mask, mask_)
        # fc layer
        x = self.fc(x)
        return x


class Transformer(nn.Module):
    def __init__(self, vocab_size, vocab_size_, pos_size=128, pad=0, pad_=0,
                 size=256, n_head=8, n_layer=6,
                 dropout=0, exp=4, device=None,):
        super().__init__()
        self.vocab_size = vocab_size
        self.vocab_size_ = vocab_size_
        self.pos_size = pos_size
        self.pad = pad
        self.size = size
        self.n_head = n_head
        self.n_layer = n_layer
        self.dropout = dropout
        self.exp = exp
        self.device = device

        self.encoder = Encoder(
                vocab_size=self.vocab_size,
                pos_size=self.pos_size,
                size=self.size,
                n_head=self.n_head,
                n_layer=self.n_layer,
                dropout=self.dropout,
                exp=self.exp,
                device=self.device)

        self.decoder = Decoder(
                vocab_size=self.vocab_size_,
                pos_size=self.pos_size,
                size=self.size,
                n_head=self.n_head,
                n_layer=self.n_layer,
                dropout=self.dropout,
                exp=self.exp,
                device=self.device)

    def forward(self, x, x_):
        # make source mask
        mask = x != self.pad
        mask = mask.unsqueeze(1).unsqueeze(2)
        # make target mask
        mask_ = torch.ones((x_.shape[1], x_.shape[1]))
        mask_ = torch.tril(mask_)
        mask_ = mask_.expand(x_.shape[0], 1, x_.shape[1], x_.shape[1])

        # encode - decode
        x = self.encoder(x, mask)
        x = self.decoder(x_, x, mask, mask_)

        return x
