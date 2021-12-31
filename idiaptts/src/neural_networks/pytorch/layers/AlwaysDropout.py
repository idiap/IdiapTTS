#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#

from torch.nn import Dropout
from torch import Tensor
import torch.nn.functional as F


class AlwaysDropout(Dropout):

    def forward(self, input: Tensor) -> Tensor:
        return F.dropout(input, self.p, True, self.inplace)
