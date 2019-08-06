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

        default_hparams = {
            "input_type": "mulaw-quantize",  # raw [-1, 1], mulaw [-1, 1], mulaw-quantize [0, mu]
            "quantize_channels": 256,
            "out_channels": 256,
            "layers": 24,  # 20
            "stacks": 4,  # 2
            "residual_channels": 512,
            "gate_channels": 512,
            "skip_out_channels": 256,
            "dropout": 0.95,
            "kernel_size": 3,
            "weight_normalization": False,
            "cin_channels": 63,
            "upsample_conditional_features": False,
            "upsample_scales": [
                5,
                4,
                2
            ],
            "freq_axis_kernel_size": 3,
            "gin_channels": -1,
            "n_speakers": 1,
            "use_speaker_embedding": False,
        }

        # Fill in missing default parameters.
        for key, value in default_hparams.items():
            if not hasattr(hparams, key) or getattr(hparams, key) is None:
                logging.warning("Hyperparamter {} of hparams is not set, using default value {} instead.".format(key, value))
                setattr(hparams, key, value)

        # Save the in to out ratio for incremental_forward.
        if hparams.upsample_conditional_features:
            self.in_out_multiplier = reduce(mul, hparams.upsample_scales, 1)
        else:
            self.in_out_multiplier = 1

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
                num_frames_to_gen = seq_lengths_inputs[0] * self.in_out_multiplier
                output = self.model.incremental_forward(c=inputs, T=num_frames_to_gen, softmax=True, quantize=True)
                # Output shape is B x C x T.

        return output, None

    def set_gpu_flag(self, use_gpu):
        self.use_gpu = use_gpu

    def init_hidden(self, batch_size=1):
        return None

    def parameters(self):
        return self.model.parameters()
