#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#

from typing import Callable

import torch
import torch.nn as nn


class ApplyFunction(nn.Module):
    def __init__(self, fn: Callable, apply_during_testing: bool = True):
        super().__init__()

        self.fn = fn
        self.apply_during_testing = apply_during_testing

    def forward(self, input_: torch.Tensor):
        if self.apply_during_testing or self.training:
            return self.fn(input_)
        else:
            return input_
