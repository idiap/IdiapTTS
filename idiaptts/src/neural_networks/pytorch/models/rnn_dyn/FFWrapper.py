#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#

import math
from numpy.lib.arraysetops import isin

import torch
import torch.nn as nn
from torch.nn.modules.activation import LeakyReLU

from .ApplyFunction import ApplyFunction
from .CustomWrapper import CustomWrapper
from .Mask import Mask
from .Norm import SigmoidNorm, LinearNorm
from .Pooling import SelectLastPooling, MeanPooling
from .VAE import VanillaVAE
from idiaptts.src.neural_networks.pytorch.layers.AlwaysDropout import AlwaysDropout


class FFWrapper(CustomWrapper):
    nonlin_options = {'relu': nn.ReLU, 'tanh': nn.Tanh}

    def __init__(self, in_dim, layer_config):
        super(FFWrapper, self).__init__()
        if "soft" in layer_config.type.lower() \
                and 'dim' not in layer_config.kwargs:
            raise ValueError("{} needs the dim parameter, otherwise it is "
                             "computed over batch and time.".format(
                                 layer_config.type))

        self._create_module(in_dim, layer_config)

    def _create_module(self, in_dim, layer_config):
        nonlin = self._get_nonlin(layer_config.nonlin)
        layer_list = []
        for _ in range(layer_config.num_layers):

            # Handle special layers here.
            if layer_config.type == "PoolLast":
                layer = SelectLastPooling(**layer_config.kwargs)
            elif layer_config.type == "PoolMean":
                layer = MeanPooling(**layer_config.kwargs)
            elif layer_config.type == "VAE" or layer_config.type == "VanillaVAE":
                layer = VanillaVAE(in_dim, layer_config.out_dim)
            elif layer_config.type == "Embedding":
                layer = nn.Embedding(**layer_config.kwargs)
                in_dim = layer_config.kwargs['embedding_dim']
            elif layer_config.type == "AlwaysDropout":
                layer = AlwaysDropout(**layer_config.kwargs)
            elif layer_config.type == "Mask":
                layer = Mask(**layer_config.kwargs)
            elif layer_config.type == "SigmoidNorm":
                layer = SigmoidNorm(**layer_config.kwargs)
            elif layer_config.type == "LinearNorm":
                layer = LinearNorm(**layer_config.kwargs)
            elif layer_config.type == "ApplyFunction":
                layer = ApplyFunction(**layer_config.kwargs)
            # Handle most PyTorch layers here.
            elif layer_config.needs_in_dim:
                if layer_config.out_dim is not None:
                    layer = getattr(torch.nn, layer_config.type)(
                        in_dim, layer_config.out_dim, **layer_config.kwargs)
                    in_dim = layer_config.out_dim
                else:
                    layer = getattr(torch.nn, layer_config.type)(
                        in_dim, **layer_config.kwargs)
            else:
                layer = getattr(torch.nn, layer_config.type)(
                    **layer_config.kwargs)
            layer_list.append(layer)

            if nonlin is not None:
                if len(layer_list) > 0:
                    self.reset_parameters(layer_list[-1], nonlin)
                layer_list.append(nonlin())

            if layer_config.dropout > 0.0:
                layer_list.append(nn.Dropout(layer_config.dropout))

        # if len(layer_list) > 1:
        self.module = nn.Sequential(*layer_list)
        # else:
        #     self.module = layer_list[0]
        self.out_dim = in_dim

    def _get_nonlin(self, nonlin: str):
        if nonlin is None:
            return None
        elif nonlin in self.nonlin_options:
            # LEGACY support for configs saved with 'relu' and 'tanh'
            return self.nonlin_options[nonlin.lower()]
        else:
            return getattr(nn, nonlin)

    @staticmethod
    def reset_parameters(layer: nn.Module, nonlin: nn.Module):
        """
        As of 10.6.21 PyTorch is still using outdated initialisation
        methods (see https://github.com/pytorch/pytorch/issues/18182).
        This function tries to apply the current best practises for
        convolutional and linear layers instead.

        :param layer: The layer to initialise the weights of.
        :type layer: nn.Module
        :param nonlin: The type of non-linearity following the layer.
        :type nonlin: nn.Module
        """
        return  # TODO: Test effect of better initialisation.

        # Only fix conv and linear layers for now.
        if not (isinstance(layer, nn.Conv1d)
                or isinstance(layer, nn.Conv2d)
                # or isinstance(layer, nn.Conv3d)  # There is probably no use case here.
                or isinstance(layer, nn.Linear)):
            return

        if nonlin == nn.ReLU or nonlin == nn.LeakyReLU:
            if nonlin == nn.LeakyReLU:
                negative_slope = nonlin.negative_slope
            else:
                negative_slope = 0

            nn.init.kaiming_normal_(layer.weight, a=negative_slope,
                                    mode='fan_out')
            if layer.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(
                    layer.weight)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(layer.bias, -bound, bound)

        elif nonlin == nn.Tanh or nonlin == nn.Sigmoid:
            nn.init.xavier_normal_(
                layer.weight,
                gain=nn.init.calculate_gain(nonlin.__name__.lower()))

    def forward(self, input_, **kwargs):
        output = super(FFWrapper, self).forward(input_, **kwargs)
        if type(self.module[0]) is nn.Embedding:
            output = output.squeeze(dim=2)
        seq_lengths_input, max_length_inputs = self._compute_seq_lengths(
            kwargs["seq_lengths_input"], kwargs["max_length_inputs"])
        kwargs["seq_lengths_input"] = seq_lengths_input
        kwargs["max_length_inputs"] = max_length_inputs
        return output, kwargs

    def _compute_seq_lengths(self, seq_lengths_input, max_length_inputs):
        for layer in self.module:
            if hasattr(layer, 'get_output_length') \
                    and callable(layer.get_output_length):
                seq_lengths_input = layer.get_output_length(seq_lengths_input)
                max_length_inputs = layer.get_output_length(max_length_inputs)
        return seq_lengths_input, max_length_inputs

    def __getitem__(self, item):
        return self.module.__getitem__(item)

    def __setitem__(self, key, value):
        return self.module.__setitem__(key, value)  # TODO: Untested
