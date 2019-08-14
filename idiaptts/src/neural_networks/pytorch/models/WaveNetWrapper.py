#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#

# System imports.
import logging
from operator import mul
from functools import reduce
import torch
import torch.nn as nn

# Third-party imports.
from wavenet_vocoder import WaveNet
from wavenet_vocoder.util import is_scalar_input

# Local source tree imports.


class WaveNetWrapper(nn.Module):
    """A wrapper around r9y9's WaveNet implementation to integrate it seamlessly into the framework."""
    IDENTIFIER = "r9y9WaveNet"

    def __init__(self, dim_in, dim_out, hparams):
        super().__init__()

        self.len_in_out_multiplier = hparams.len_in_out_multiplier

        # Use the wavenet_vocoder builder to create the model.
        self.model = WaveNet(out_channels=hparams.out_channels,
                             layers=hparams.layers,
                             stacks=hparams.stacks,
                             residual_channels=hparams.residual_channels,
                             gate_channels=hparams.gate_channels,
                             skip_out_channels=hparams.skip_out_channels,
                             kernel_size=hparams.kernel_size,
                             dropout=hparams.dropout,
                             weight_normalization=hparams.weight_normalization,
                             cin_channels=hparams.cin_channels,
                             gin_channels=hparams.gin_channels,
                             n_speakers=hparams.n_speakers,
                             upsample_conditional_features=hparams.upsample_conditional_features,
                             upsample_scales=hparams.upsample_scales,
                             freq_axis_kernel_size=hparams.freq_axis_kernel_size,
                             scalar_input=is_scalar_input(hparams.input_type),
                             use_speaker_embedding=hparams.use_speaker_embedding,
                             )

    def forward(self, inputs, hidden, seq_lengths_inputs, max_length_inputs, target=None, seq_lengths_target=None):

        if target is not None:  # During training and testing with teacher forcing.
            output = self.model(target, c=inputs, g=None, softmax=False)
            # output = self.model(target, c=inputs[:, :, :target.shape[2]], g=None, softmax=False)
            # Output shape is B x C x T. Don't permute here because CrossEntropyLoss requires the same shape.
        else:  # During inference.
            with torch.no_grad():
                self.model.make_generation_fast_()
                num_frames_to_gen = seq_lengths_inputs[0] * self.len_in_out_multiplier
                output = self.model.incremental_forward(c=inputs, T=num_frames_to_gen, softmax=True, quantize=True)
                # Output shape is B x C x T.

        return output, None

    def set_gpu_flag(self, use_gpu):
        self.use_gpu = use_gpu

    def init_hidden(self, batch_size=1):
        return None

    def parameters(self):
        return self.model.parameters()
