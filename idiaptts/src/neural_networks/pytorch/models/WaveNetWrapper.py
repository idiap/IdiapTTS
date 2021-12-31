#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#

# System imports.
import copy
from types import MethodType

# Third-party imports.
import numpy as np
import torch
import torch.nn as nn
from torch.nn import LayerNorm
from torch.nn.utils.weight_norm import WeightNorm
from wavenet_vocoder import WaveNet
from wavenet_vocoder.modules import ConvTranspose2d, Conv1d
from wavenet_vocoder.util import is_scalar_input

# Local source tree imports.


class WaveNetWrapper(nn.Module):
    """A wrapper around r9y9's WaveNet implementation to integrate it seamlessly into the framework."""
    IDENTIFIER = "r9y9WaveNet"

    class Config:
        INPUT_TYPE_MULAW = "mulaw-quantize"
        INPUT_TYPE_RAW = "raw"

        def __init__(
                self,
                cin_channels=80,
                dropout=0.05,
                freq_axis_kernel_size=3,
                gate_channels=512,
                gin_channels=-1,
                hinge_regularizer=True,  # Only used in MoL prediction (INPUT_TYPE_RAW).
                kernel_size=3,
                layers=24,
                log_scale_min=float(np.log(1e-14)),  # Only used in INPUT_TYPE_RAW.
                n_speakers=1,
                out_channels=256,  # Use num_mixtures * 3 (pi, mean, log_scale) for INPUT_TYPE_RAW.
                residual_channels=512,
                scalar_input=is_scalar_input(INPUT_TYPE_MULAW),
                skip_out_channels=256,
                stacks=4,
                upsample_conditional_features=False,
                upsample_scales=[5, 4, 2],
                use_speaker_embedding=False,
                weight_normalization=True,
                legacy=False):

            self.cin_channels = cin_channels
            self.dropout = dropout
            self.freq_axis_kernel_size = freq_axis_kernel_size
            self.gate_channels = gate_channels
            self.gin_channels = gin_channels
            self.hinge_regularizer = hinge_regularizer
            self.kernel_size = kernel_size
            self.layers = layers
            self.log_scale_min = log_scale_min
            self.n_speakers = n_speakers
            self.out_channels = out_channels
            self.residual_channels = residual_channels
            self.scalar_input = scalar_input
            self.skip_out_channels = skip_out_channels
            self.stacks = stacks
            self.upsample_conditional_features = upsample_conditional_features
            self.upsample_scales = upsample_scales
            self.use_speaker_embedding = use_speaker_embedding
            self.weight_normalization = weight_normalization
            self.legacy = legacy

        def create_model(self):
            return WaveNetWrapper(self)

    def __init__(self, config):
        super().__init__()

        # self.len_in_out_multiplier = hparams.len_in_out_multiplier

        # Use the wavenet_vocoder builder to create the model.
        self.model = WaveNet(
            out_channels=config.out_channels,
            layers=config.layers,
            stacks=config.stacks,
            residual_channels=config.residual_channels,
            gate_channels=config.gate_channels,
            skip_out_channels=config.skip_out_channels,
            kernel_size=config.kernel_size,
            dropout=config.dropout,
            weight_normalization=config.weight_normalization,
            cin_channels=config.cin_channels,
            gin_channels=config.gin_channels,
            n_speakers=config.n_speakers,
            upsample_conditional_features=config.upsample_conditional_features,
            upsample_scales=config.upsample_scales,
            freq_axis_kernel_size=config.freq_axis_kernel_size,
            scalar_input=config.scalar_input,
            use_speaker_embedding=config.use_speaker_embedding,
            legacy=config.legacy
        )

        self.has_weight_norm = True
        # self.__deepcopy__ = MethodType(__deepcopy__, self)

    def forward(self, input_, target, seq_lengths, *_):

        if target is not None:  # During training and testing with teacher forcing.
            assert self.has_weight_norm, "Model has been used for generation " \
                "and weight norm was removed, cannot continue training. Remove"\
                " the make_generation_fast_() call to continue training after" \
                " generation."
            output = self.model(target, c=input_, g=None, softmax=False)
            # output = self.model(target, c=inputs[:, :, :target.shape[2]], g=None, softmax=False)
            # Output shape is B x C x T. Don't permute here because CrossEntropyLoss requires the same shape.
        else:  # During inference.
            with torch.no_grad():
                self.model.make_generation_fast_()  # After calling this the training cannot be continued.
                self.has_weight_norm = False
                assert(len(seq_lengths) == 1), "Batch synth is not supported."
                num_frames_to_gen = seq_lengths[0] * self.len_in_out_multiplier
                output = self.model.incremental_forward(
                    c=input_, T=num_frames_to_gen, softmax=True, quantize=True)
                # output = self.model.incremental_forward(
                #   c=inputs[:, :, :1000], T=torch.tensor(1000), softmax=True, quantize=True)

        # Output shape is B x C x T.
        return output, None

    def set_gpu_flag(self, use_gpu):
        self.use_gpu = use_gpu

    def init_hidden(self, batch_size=1):
        return None

    def parameters(self):
        return self.model.parameters()


# def __deepcopy__(self, memo):
#     """
#     Fix the deepcopy operation with WeigthNorm layers by removing all
#     during copying. The code was posted as a solution at
#     https://github.com/pytorch/pytorch/issues/28594
#     """
#     # save and delete all weightnorm weights on self
#     weights = {}
#     for hook in self._forward_pre_hooks.values():
#         if isinstance(hook, WeightNorm):
#             weights[hook.name] = getattr(self, hook.name)
#             delattr(self, hook.name)

#     # remove this deepcopy method, restoring the object's original one if necessary
#     __deepcopy__ = self.__deepcopy__
#     if orig_deepcopy:
#         self.__deepcopy__ = orig_deepcopy
#     else:
#         del self.__deepcopy__
#     # actually do the copy
#     result = copy.deepcopy(self)

#     # restore weights and method on self
#     for name, value in weights.items():
#         setattr(self, name, value)
#     self.__deepcopy__ = __deepcopy__

#     return result

for layer in [Conv1d, ConvTranspose2d]:
    orig_deepcopy = getattr(layer, '__deepcopy__', None)
    def __deepcopy__(self, memo):
        """
        Fix the deepcopy operation with WeigthNorm layers by removing all
        during copying. The code was posted as a solution at
        https://github.com/pytorch/pytorch/issues/28594
        """
        # save and delete all weightnorm weights on self
        weights = {}
        for hook in self._forward_pre_hooks.values():
            if isinstance(hook, WeightNorm):
                weights[hook.name] = getattr(self, hook.name)
                delattr(self, hook.name)

        # remove this deepcopy method, restoring the object's original one if necessary
        __deepcopy__ = self.__deepcopy__
        if orig_deepcopy:
            self.__deepcopy__ = orig_deepcopy
        else:
            del self.__deepcopy__
        # actually do the copy
        result = copy.deepcopy(self)

        # restore weights and method on self
        for name, value in weights.items():
            setattr(self, name, value)
        self.__deepcopy__ = __deepcopy__

        return result
    # bind __deepcopy__ to the weightnorm'd layer
    # layer.__deepcopy__ = __deepcopy__.__get__(layer, layer.__class__)
    layer.__deepcopy__ = MethodType(__deepcopy__, layer)
