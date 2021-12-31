#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#

from typing import List, Tuple
from idiaptts.src.neural_networks.pytorch.models.rnn_dyn.FFWrapper import FFWrapper

import torch
import torch.nn as nn

from .TransposingWrapper import TransposingWrapper


class CNNWrapper(TransposingWrapper):
    def __init__(self, in_dim, layer_config, batch_first: bool = True):
        super(CNNWrapper, self).__init__(in_dim, layer_config, batch_first)

    def _create_module(self, in_dim, layer_config):
        layer_list = []
        nonlin = self._get_nonlin(layer_config.nonlin)
        for _ in range(layer_config.num_layers):
            if 'padding' not in layer_config.kwargs \
                    and ('stride' not in layer_config.kwargs
                         or layer_config.kwargs['stride'] == 1) \
                    and ('dilation' not in layer_config.kwargs
                         or layer_config.kwargs['dilation'] == 1):

                kernel = layer_config.kwargs.get('kernel_size')
                if type(kernel) in [Tuple, List]:
                    kernel = kernel[0]
                # Designed so that sequence length remains the same.
                # Usefull for speech.
                # TODO: Take into account dilation and padding.
                layer_config.kwargs['padding'] = int((kernel - 1) / 2)

            layer = getattr(torch.nn, layer_config.type)(
                in_dim, layer_config.out_dim, **layer_config.kwargs)
            layer_list.append(layer)
            in_dim = layer_config.out_dim

            # From NVIDIA's Tacotron2 implementation.
            # TODO: Make this an else when switching to new initialisation.
            torch.nn.init.xavier_uniform_(
                layer.weight,
                gain=torch.nn.init.calculate_gain(layer_config.type.lower()))

            if nonlin is not None:
                if len(layer_list) > 0:
                    FFWrapper.reset_parameters(layer_list[-1], nonlin)
                layer_list.append(nonlin())

        self.module = nn.Sequential(*layer_list)
        self.out_dim = in_dim

    def forward(self, input, **kwargs):
        output, kwargs = super(CNNWrapper, self).forward(input, **kwargs)

        kwargs["seq_lengths_input"] = self.get_output_length(
            kwargs['seq_lengths_input'])
        kwargs["max_length_inputs"] = self.get_output_length(
            kwargs['max_length_inputs'])
        return output, kwargs

    def get_output_length(self, seq_lengths_input):
        for layer in self.module:
            if hasattr(layer, "padding"):
                padding = layer.padding
                if type(padding) in [tuple, list]:
                    padding = padding[0]
                dilation = layer.dilation
                if type(dilation) in [tuple, list]:
                    dilation = dilation[0]
                kernel_size = layer.kernel_size
                if type(kernel_size) in [tuple, list]:
                    kernel_size = kernel_size[0]
                stride = layer.stride
                if type(stride) in [tuple, list]:
                    stride = stride[0]

                # Formula: https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html#torch.nn.Conv1d
                seq_lengths_input = (seq_lengths_input + 2 * padding
                                     - dilation * (kernel_size - 1) - 1) \
                                    // stride + 1

        return seq_lengths_input
