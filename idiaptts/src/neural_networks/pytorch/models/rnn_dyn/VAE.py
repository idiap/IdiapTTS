#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#

import torch
import torch.nn as nn


class VanillaVAE(nn.Module):
    def __init__(self, dim_in, latent_dim):
        super(VanillaVAE, self).__init__()
        self.linear = nn.Linear(dim_in, latent_dim * 2, bias=False)

    def forward(self, input):
        hidden = self.linear(input)
        mu, log_var = torch.split(hidden, hidden.shape[2] // 2, dim=2)
        z = self._reparametrize(mu, log_var)
        return z, mu, log_var

    def _reparametrize(self, mu: torch.Tensor, log_var: torch.Tensor) \
            -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu
