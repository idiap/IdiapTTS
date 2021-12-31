#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#

import torch
import torch.nn as nn


class SigmoidNorm(nn.Module):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, input_: torch.Tensor):
        input_sig = torch.sigmoid(input_)
        return input_sig / (input_sig.sum(dim=self.dim, keepdim=True))


class LinearNorm(nn.Module):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, input_: torch.Tensor):
        return input_ / input_.sum(dim=self.dim, keepdim=True)
