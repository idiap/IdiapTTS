#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#

import torch.nn as nn


class Pooling(nn.Module):
    def __init__(self, batch_first):
        super().__init__()
        self.batch_first = batch_first

    def extra_repr(self):
        return "batch_first={}".format(self.batch_first)

    def get_output_length(self, seq_lengths_input):
        return seq_lengths_input.fill_(1)

    def select_inputs(self, input_, **kwargs):
        return input_, kwargs.pop("seq_lengths_input", None)


class SelectLastPooling(Pooling):
    def __init__(self, batch_first):
        super(SelectLastPooling, self).__init__(batch_first)

    def forward(self, input_):
        input_, lengths = input_
        batch_dim = 0 if self.batch_first else 1
        batch_size = input_.shape[batch_dim]
        if lengths is None:
            time_dim = 1 if self.batch_first else 0
            seq_len_indices = [input_.shape[time_dim] - 1] * batch_size
        else:
            seq_len_indices = [length - 1 for length in lengths]
        batch_indices = [i for i in range(batch_size)]

        if self.batch_first:
            return input_[batch_indices, seq_len_indices].unsqueeze(dim=1)
        else:
            return input_[seq_len_indices, batch_indices].unsqueeze(dim=0)


class MeanPooling(Pooling):
    def __init__(self, batch_first):
        super().__init__(batch_first)
        self.time_dim = 1 if batch_first else 0

    def forward(self, input_):
        input_, lengths = input_

        input_sum = input_.sum(self.time_dim, keepdim=True)

        batch_dim = len(lengths)
        missing_dims = [1] * max(0, input_sum.ndim - 2)
        if self.batch_first:
            lengths = lengths.view(batch_dim, 1, *missing_dims).float()
        else:
            lengths = lengths.view(1, batch_dim, *missing_dims).float()

        input_mean = input_sum / lengths
        return input_mean
