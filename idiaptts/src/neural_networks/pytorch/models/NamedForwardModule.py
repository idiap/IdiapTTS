#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#

from functools import reduce, partial
from typing import Union, Any, List, Optional, cast, Dict
import logging

import torch

from idiaptts.src.neural_networks.pytorch.models.ModelConfig import ModelConfig


class NamedForwardModule(torch.nn.ModuleList):

    def __init__(self,
                 input_names: List[str],
                 batch_first: bool,
                 input_merge_type: str = ModelConfig.MERGE_TYPE_CAT,
                 name: str = None,
                 output_names: List[str] = None,  # TODO: Use config: ModelConfig as input.
                 teacher_forcing_input_names: List[str] = None):
        super().__init__()
        self.input_names = input_names
        self.batch_first = batch_first
        self.input_merge_type = input_merge_type
        self.name = name
        self.output_names = output_names
        assert teacher_forcing_input_names is None \
            or len(teacher_forcing_input_names) > 0, "Empty list given not " \
            "allowed for teacher_forcing_input_names, use None instead."
        self.teacher_forcing_input_names = teacher_forcing_input_names

    @property
    def is_autoregressive(self):
        return self.teacher_forcing_input_names is not None

    def forward(self, data, lengths, max_lengths, **kwargs):
        input_ = NamedForwardModule._get_inputs(
            data, self.input_names, self.input_merge_type, self.batch_first)

        if self.is_autoregressive:
            # At inference time teacher forcing inputs are missing.
            if all([name in data for name in self.teacher_forcing_input_names]):
                target = NamedForwardModule._get_inputs(
                    data, self.teacher_forcing_input_names,
                    ModelConfig.MERGE_TYPE_CAT, batch_first=True)
            else:
                target = None
            output = self.forward_module(input_, lengths, max_lengths,
                                         target=target, **kwargs)
        else:
            output = self.forward_module(input_, lengths, max_lengths, **kwargs)

        output_dict = self._map_to_output_names(output)
        data.update(output_dict)

    def inference(self, data, lengths, max_lengths, *args, **kwargs):
        if self.is_autoregressive:
            data_filtered = {k: v for k, v in data.items()
                             if k not in self.teacher_forcing_input_names}
            lengths_filtered = {k: v for k, v in lengths.items()
                                if k not in self.teacher_forcing_input_names}
            max_lengths_filtered = {k: v for k, v in max_lengths.items()
                                    if k not in self.teacher_forcing_input_names}

            self.forward(data_filtered, lengths_filtered, max_lengths_filtered,
                         *args, **kwargs)

            data.update(data_filtered)
            lengths.update(lengths_filtered)
            max_lengths.update(max_lengths_filtered)
        else:
            self.forward(data, lengths, max_lengths, *args, **kwargs)

    @staticmethod
    def _get_inputs(data: Dict, input_names: Union[str, List[str]],
                    input_merge_type: str, batch_first: bool,
                    return_lengths: bool = False):
        try:
            if type(input_names) is not str:
                inputs = [data[name] for name in input_names]
            else:
                inputs = [data[input_names]]
        except KeyError as e:
            logging.error("Cannot find {} in input data.".format(e.args))
            raise
        time_dim = 1 if batch_first else 0
        lengths = [i.shape[time_dim] for i in inputs if len(i.shape) > 2]
        max_length = max(lengths) if len(lengths) > 0 else 1
        inputs = [NamedForwardModule._broadcast_time_dim(i, max_length,
                                                         batch_first)
                  for i in inputs]

        if len(inputs) > 1:
            try:
                output = NamedForwardModule.merge(inputs, input_merge_type,
                                                  batch_first)
            except RuntimeError:
                logging.error("Failed to merge {}. Did you specify "
                              "batch_first=False/True everywhere?".format(
                                  ", ".join(input_names)))
                raise
        else:
            output = inputs[0]

        if return_lengths:
            return output, lengths
        else:
            return output

    @staticmethod
    def merge(inputs: List, merge_type: str, batch_first: bool):
        # TODO: Should MERGE_TYPEs be defined in this class instead?
        if merge_type == ModelConfig.MERGE_TYPE_CAT:
            # Default behaviour.
            output = torch.cat(inputs, dim=2)
        elif merge_type == ModelConfig.MERGE_TYPE_ADD:
            output = torch.sum(torch.stack(inputs), dim=0)
        elif merge_type == ModelConfig.MERGE_TYPE_MEAN:
            # Used for mean pooling.
            output = torch.mean(torch.stack(inputs), dim=0)
        elif merge_type == ModelConfig.MERGE_TYPE_MUL:
            output = reduce(lambda x, y: x * y, inputs)
        elif merge_type == ModelConfig.MERGE_TYPE_ATTENTION:
            output = reduce(lambda x, y: x * y, inputs)
            output = output.sum(dim=1 if batch_first else 0,
                                keepdim=True)
        else:
            # Basically MERGE_TYPE_LIST. Used to group intermediate
            # outputs, which are then given to modules expecting
            # multiple inputs.
            output = inputs
        return output

    @staticmethod
    def _broadcast_time_dim(tensor, length, batch_first):
        time_dim = 1 if batch_first else 0
        if tensor.ndim < 3:
            tensor = tensor.unsqueeze(time_dim)
        if tensor.shape[time_dim] == 1:
            if batch_first:
                tensor = tensor.repeat((1, length, 1))
            else:
                tensor = tensor.repeat((length, 1, 1))
        return tensor

    def forward_module(self, input_, lengths, max_lengths):
        raise NotImplementedError()

    def _map_to_output_names(self, output):
        if type(output) not in (tuple, list):
            output = (output,)
        if len(self.output_names) != len(output):
            raise ValueError(
                "{} output name(s) are defined but {} returns {} outputs."
                .format(len(self.output_names),
                        self.name if self.name is not None else type(self),
                        len(output)))
        return {name: value for name, value in zip(self.output_names, output)}
