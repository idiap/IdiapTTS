#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#

from typing import Union, Any, List, Optional, cast, Dict

import torch

from idiaptts.src.neural_networks.pytorch.models.NamedForwardModule import NamedForwardModule
from idiaptts.src.neural_networks.pytorch.models.ModelConfig import ModelConfig


class NamedLoss(NamedForwardModule):

    class Config(ModelConfig):
        def __init__(self,
                     name: str,
                     type_: str,
                     seq_mask: str,  # Required, but can None.
                     input_names: List[str],
                     batch_first: bool = True,
                     input_merge_type: str = ModelConfig.MERGE_TYPE_LIST,
                     squeeze_inputs: bool = False,  # Used for classifications.
                     start_step: int = 0,
                     reduction: str = 'mean_per_frame',
                     loss_weight: float = 1.0,
                     **kwargs):
            super().__init__(input_names=input_names, batch_first=batch_first,
                             input_merge_type=input_merge_type, name=name,
                             **kwargs)

            self.type = type_
            self.seq_mask = seq_mask
            self.squeeze_inputs = squeeze_inputs
            self.start_step = start_step  # TODO: Use it.
            if self.seq_mask is None and (reduction == 'mean_per_frame'
                                          or reduction == 'mean_per_sample'):
                self.reduction = 'mean'
            else:
                self.reduction = reduction
            self.loss_weight = loss_weight

        def create_loss(self):
            return NamedLoss(self)

    def __init__(self, config: Config):
        super().__init__(input_names=config.input_names,
                         batch_first=config.batch_first,
                         input_merge_type=config.input_merge_type,
                         name=config.name,
                         output_names=config.output_names)
        self.seq_mask = config.seq_mask
        self.squeeze_inputs = config.squeeze_inputs
        self.start_step = config.start_step
        self.reduction = config.reduction
        self.loss_weight = config.loss_weight

        if not hasattr(self, "loss_fn") or self.loss_fn is None:
            try:
                import idiaptts.src.neural_networks.pytorch.loss as losses
                self.loss_fn = getattr(losses, config.type)(reduction='none',
                                                            **config.kwargs)
            except AttributeError:
                self.loss_fn = getattr(torch.nn, config.type)(reduction='none',
                                                              **config.kwargs)

    def forward(self, data, length_dict, step):
        inputs = NamedForwardModule._get_inputs(
            data=data,
            input_names=self.input_names,
            batch_first=self.batch_first,
            input_merge_type=ModelConfig.MERGE_TYPE_LIST)

        if self.squeeze_inputs:
            batch_dim = 0 if self.batch_first else 1
            batch_size = inputs[0].shape[batch_dim]
            inputs = [input_.squeeze() for input_ in inputs]
            # Handle a special case where the batch_dim is squeezed.
            if batch_size == 1:
                for idx, input_ in enumerate(inputs):
                    # Omit single dimensional outputs as they are class labels. Probably?
                    # if input_.ndim > 0 and input_.shape[0] != 1:
                    inputs[idx] = input_.unsqueeze(batch_dim)

        if type(inputs) in [list, tuple]:
            loss_dict = self._map_to_output_names(self.loss_fn(*inputs))
        else:
            loss_dict = self._map_to_output_names(self.loss_fn(inputs))  # Starred expression would split tensor.

        if self.seq_mask is not None:
            seq_mask = NamedForwardModule._get_inputs(
                data=data,
                input_names=[self.seq_mask],
                batch_first=self.batch_first,
                input_merge_type=ModelConfig.MERGE_TYPE_LIST
            )
            loss_dict = {k: seq_mask * v for k, v in loss_dict.items()}

        # loss_dict = self._reduce(loss_dict, length_dict)
        loss_dict = {k: self.apply_weight(self._reduce(v, length_dict), step)
                     for k, v in loss_dict.items()}

        data.update(loss_dict)
        return loss_dict

    def apply_weight(self, loss, step):
        weight = 0. if step < self.start_step else self.loss_weight
        return loss * weight

    def _reduce(self, v, length_dict):
        if self.reduction == 'mean_per_frame':
            batch_time_sum = v.sum(dim=(0, 1))
            total_num_frames = sum(length_dict[self.seq_mask]).float()
            return (batch_time_sum / total_num_frames).mean()
        elif self.reduction == 'mean_per_sample':
            time_dim = 1 if self.batch_first else 0
            time_sum = v.sum(dim=time_dim)
            batch_lengths = length_dict[self.seq_mask].unsqueeze(-1).float()
            return (time_sum / batch_lengths).mean()
        elif self.reduction == 'mean':
            return v.mean()
        elif self.reduction == 'sum':
            return v.sum()
        elif self.reduction == 'none':
            return v
        else:
            raise NotImplementedError("Unknown reduction type {}.".format(
                self.reduction))
