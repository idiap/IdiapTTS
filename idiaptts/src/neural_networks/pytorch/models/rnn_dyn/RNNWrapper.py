#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#

from typing import List

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .Config import Config
from .CustomWrapper import CustomWrapper


class RNNWrapper(CustomWrapper):
    """
    Wraps an RNN module by packing and unpacking the sequence. Also
    stores the hidden state internally.
    """
    def __init__(self, in_dim: int, layer_config: List[Config],
                 batch_first: bool = True, enforce_sorted: bool = True):

        if layer_config.nonlin is not None and layer_config.type != 'RNN':
            raise NotImplementedError("Non-linearity is not supported for {}."
                                      .format(layer_config.type))

        super(RNNWrapper, self).__init__()

        self.batch_first = batch_first
        self.enforce_sorted = enforce_sorted
        self.bidirectional = layer_config.kwargs.get("bidirectional", False)
        self.hidden = None
        self.pack = True
        self.unpack = True

        self._create_module(in_dim, layer_config)

        self.out_dim = layer_config.out_dim
        if self.bidirectional:
            self.out_dim *= 2

    def _create_module(self, in_dim, layer_config):
        if layer_config.type == 'RNN':
            layer_config.kwargs['nonlinearity'] = layer_config.nonlin
        self.module = getattr(torch.nn, layer_config.type)(
            input_size=in_dim,
            hidden_size=layer_config.out_dim,
            num_layers=layer_config.num_layers,
            dropout=layer_config.dropout,
            batch_first=self.batch_first,
            **layer_config.kwargs)
        # TODO: PyTorch's uniform initialization of RNN weights is good?
        self._create_hidden_states(layer_config)

    def _create_hidden_states(self, layer_config):
        num_directions = 2 if self.bidirectional else 1

        hidden_init_value = layer_config.kwargs.get('hidden_init_value', 0.0)
        h0_init = torch.Tensor(layer_config.num_layers * num_directions, 1,
                               layer_config.out_dim).fill_(hidden_init_value)
        c0_init = torch.Tensor(layer_config.num_layers * num_directions, 1,
                               layer_config.out_dim).fill_(hidden_init_value)

        if layer_config.kwargs.get('train_hidden_init', False):
            self.register_parameter('h_0', nn.Parameter(h0_init))
            self.register_parameter('c_0', nn.Parameter(c0_init))
        else:
            # TODO: add persistent=False so that they are not in state dict
            self.register_buffer('h_0', h0_init)
            self.register_buffer('c_0', c0_init)

    def init_hidden(self, batch_size=1):
        hidden_size = list(self.h_0.size())
        hidden_size[1] = batch_size  # h_0 and c_0 are always with batch dim as second dim.
        h_0 = self.h_0.expand(hidden_size).contiguous()

        if type(self.module) is torch.nn.LSTM:
            c_0 = self.c_0.expand(hidden_size).contiguous()
            self.hidden = h_0, c_0
        else:
            self.hidden = h_0

    def forward(self, input_, seq_lengths_input, max_length_inputs,
                hidden=None, **kwargs):

        if len(seq_lengths_input) > 1 and self.pack:
            input_ = pack_padded_sequence(input_,
                                          lengths=seq_lengths_input,
                                          batch_first=self.batch_first,
                                          enforce_sorted=self.enforce_sorted)

        # If self.hidden is not set here, init_hidden was not called.
        output, self.hidden = self.module(
            input_, self.hidden if hidden is None else hidden)

        if len(seq_lengths_input) > 1 and self.unpack:
            output, _ = pad_packed_sequence(output,
                                            batch_first=self.batch_first,
                                            total_length=max_length_inputs)

        kwargs["hidden"] = self.hidden
        kwargs["seq_lengths_input"] = seq_lengths_input
        kwargs["max_length_inputs"] = max_length_inputs
        return output, kwargs
