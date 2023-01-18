#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch


class Adam_impl(torch.optim.Optimizer):
    def __init__(self,
                 params,
                 lr,
                 betas=(0.9, 0.999),
                 eps=1e-8,
                 weight_decay=0,
                 maximize=False,
                 ):

        self.params = params
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.maximize = maximize
        self.defaults = {
                'lr': self.lr,
                'betas': betas,
                'eps': eps,
                'weight_decay': weight_decay,
                'maximize': maximize,
                }
        super().__init__(params, self.defaults)

    def get_lr(self, state, group):
        return group['lr']

    def init(self, state, group, param):
        """
        init
        """
        state['step'] = 0
        state['m_t'] = torch.zeros(param.shape)  # exp avg
        state['v_t'] = torch.zeros(param.shape)  # exp square avg

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:

                if p.grad is None:
                    continue

                grad = p.grad.data
                grad = -grad if self.maximize else grad
                state = self.state[p]

                if len(state) == 0:
                    self.init(state, g, p)

                beta1, beta2 = g['betas']
                p.data *= (1.0 - g['lr'] * g['weight_decay'])

                state['m_t'] = state['m_t'] * beta1 + grad*(1 - beta1)
                state['v_t'] = state['v_t'] * beta2 + grad * grad * (1 - beta2)

                state['step'] += 1
                bias_corr_1 = 1 - beta1 ** state['step']
                bias_corr_2 = 1 - beta2 ** state['step']

                lr = self.get_lr(state, g)
                p.data -= lr / bias_corr_1 * state['m_t'] * ((bias_corr_2)**0.5 +g['eps']) / (state['v_t'].sqrt())
