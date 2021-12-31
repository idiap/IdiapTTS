#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#

from typing import List

import torch
import torch.nn as nn


class Mask(nn.Module):
    def __init__(self, invert_mask: bool = True, mask_value: float = 0.0) \
            -> None:
        super().__init__()
        self.invert_mask = invert_mask
        self.mask_value = float(mask_value)

    def forward(self, input_: List[torch.Tensor]):
        input_, mask = input_
        mask = mask.bool()
        if self.invert_mask:
            mask = ~mask
        return input_.masked_fill(mask, self.mask_value)
