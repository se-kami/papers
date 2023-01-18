#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import nn


class PatchEmbed(nn.Module):
    def __init__(self, in_channels, out_channels, size):
        super().__init__()
        self.layer = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=size,
                stride=size,)

    def forward(self, x):
        x = self.layer(x)
        # height, channel, batch, channel
        x = x.permute(2, 3, 0, 1)
        # patch
        x = x.view(-1, x.shape[2], x.shape[3])
        # batch, patch, channel
        x = x.permute(1, 0, 2)
        return x


class PositionEmbed(nn.Module):
    def __init__(self, img_size, patch_size, out_size):
        super().__init__()
        n_patches = (img_size//patch_size)*(img_size//patch_size)
        dim = torch.zeros(n_patches + 1, out_size).unsqueeze(0)
        self.pos = nn.Parameter(dim, requires_grad=True)

    def forward(self, x):
        # pos = self.layer[:x.shape[0]]
        x += self.pos
        return x


class Tokenize(nn.Module):
    def __init__(self, size):
        super().__init__()
        dim = torch.zeros(size).unsqueeze(0).unsqueeze(0)
        self.token = nn.Parameter(dim, requires_grad=True)

    def forward(self, x):
        size = x.shape[0]
        token = self.token.expand(size, -1, -1)
        x = torch.cat((token, x), dim=1)
        return x


class LinearNet(nn.Module):
    def __init__(self, n_in, n_hidden, n_out):
        super().__init__()
        net = []
        net.append(nn.Linear(n_in, n_hidden))
        net.append(nn.ReLU())
        net.append(nn.Linear(n_hidden, n_out))
        net = nn.Sequential(*net)
        self.net = net

    def forward(self, x):
        x = self.net(x)
        return x


class Attention(nn.Module):
    def __init__(self, size, n_head):
        super().__init__()
        self.size = size
        self.n_head = n_head
        self.discount = (size // n_head)**(0.5)
        self.transform = nn.Linear(size, size * 3)
        self.fc = nn.Linear(size, size)

    def forward(self, x):
        n_samples, n_tokens, dim = x.shape

        x = self.transform(x)
        x = x.reshape(x.shape[0], x.shape[1], 3, self.n_head, -1)
        x = x.permute(2, 0, 3, 1, 4)  # 3, batch, heads, patches, head size

        # each one is (batch, heads, patches, head size)
        qs = x[0]
        keys = x[1].transpose(-2, -1)  # batch, heads, head size, patches
        vals = x[2]

        x = (qs @ keys) / self.discount
        attn = torch.softmax(x, dim=-1)
        x = attn @ vals
        x = x.transpose(1, 2)  # batch, patches, heads, head size
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = self.fc(x)
        return x


class Block(nn.Module):
    def __init__(self, size, n_head, exp=4, eps=1e-8):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(size, eps=eps)
        self.layer_norm_2 = nn.LayerNorm(size, eps=eps)
        self.attn = Attention(size, n_head=n_head)
        self.fc = LinearNet(size, size*exp, size)

    def forward(self, x):
        x_ = self.layer_norm_1(x)
        x_ = self.attn(x)
        x = x + x_
        x_ = self.layer_norm_2(x)
        x_ = self.fc(x)
        x = x + x_
        return x


class Classifier(nn.Module):
    def __init__(self, size, n_out, eps=1e-8):
        super().__init__()
        self.layer_norm = nn.LayerNorm(size, eps=eps)
        self.fc = nn.Linear(size, n_out)

    def forward(self, x):
        x = self.layer_norm(x)
        x = x[:, 0]
        x = self.fc(x)
        return x


class VisionTransformer(nn.Module):
    def __init__(self,
                 img_size=384,
                 size=16,
                 in_channels=3,
                 out_channels=768,
                 n_blocks=12,
                 n_head=12,
                 exp=4,
                 n_out=1000,
                 ):
        super().__init__()

        # init
        self.img_size = img_size
        self.size = size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_blocks = n_blocks
        self.n_head = n_head
        self.exp = exp
        self.n_out = n_out

        self.patch_embed = PatchEmbed(self.in_channels, self.out_channels, self.size)
        self.tokenize = Tokenize(self.out_channels)
        self.pos_embed = PositionEmbed(self.img_size, self.size, self.out_channels)

        l = []
        for i in range(self.n_blocks):
            l.append(Block(self.out_channels, n_head=self.n_head, exp=self.exp))
        self.blocks = nn.Sequential(*l)

        self.classifier = Classifier(self.out_channels, self.n_out)

    def forward(self, x):
        b_size = x.shape[0]
        x = self.patch_embed(x)
        x = self.tokenize(x)
        x = self.pos_embed(x)
        x = self.blocks(x)
        x = self.classifier(x)

        return x
