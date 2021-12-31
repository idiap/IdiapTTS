#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#

from typing import Union, Any, List, Optional, cast, Dict
import copy
import logging

import jsonpickle

from idiaptts.src.neural_networks.pytorch.models.NamedForwardModule import NamedForwardModule
from idiaptts.src.neural_networks.pytorch.models.ModelConfig import ModelConfig
import idiaptts.src.neural_networks.pytorch.models.rnn_dyn as rnn_dyn


class NamedForwardWrapper(NamedForwardModule):

    class Config(ModelConfig):
        def __init__(self,
                     wrapped_model_config,
                     input_names: List[str],
                     batch_first: bool,
                     input_merge_type: str = ModelConfig.MERGE_TYPE_CAT,
                     name: str = None,
                     output_names: List[str] = None,
                     teacher_forcing_input_names: List[str] = None):

            super().__init__(input_names=input_names, batch_first=batch_first,
                             input_merge_type=input_merge_type, name=name,
                             output_names=output_names)

            self.wrapped_model_config = wrapped_model_config
            self.teacher_forcing_input_names = teacher_forcing_input_names

        def create_model(self):
            return NamedForwardWrapper(self)

    def __init__(self, config: Config):
        super().__init__(
            input_names=config.input_names,
            input_merge_type=config.input_merge_type,
            name=config.name,
            output_names=config.output_names,
            batch_first=config.batch_first,
            teacher_forcing_input_names=config.teacher_forcing_input_names
            if hasattr(config, "teacher_forcing_input_names") else None  # Legacy support
        )

        self.config = copy.deepcopy(config)

        if config.wrapped_model_config is not None:
            self.model = config.wrapped_model_config.create_model()
        else:
            self.model = None

    def forward_module(self, input_, lengths, max_lengths, **kwargs):
        if type(self.model) is rnn_dyn.RNNDyn:
            # TODO: Rename to lengths, and max_lengths
            output, kwargs = self.model(
                input_,
                seq_lengths_input=lengths[self.input_names[0]],
                max_length_inputs=max_lengths[self.input_names[0]],
                **kwargs)
            lengths.update({output_name: kwargs['seq_lengths_input']
                            for output_name in self.output_names})
            max_lengths.update({output_name: kwargs['max_length_inputs']
                               for output_name in self.output_names})
        else:
            output = self.model(
                input_,
                lengths=lengths[self.input_names[0]],
                max_length=max_lengths[self.input_names[0]],
                **kwargs)

            if type(output) in [tuple, list]:
                # Usually the output of the model should contain a dictionary
                # with updated lengths and max_length. However, the wrapper can
                # be used with models which don't follow this procedure as well.
                output, kwargs = output
                lengths.update({output_name: kwargs['lengths']
                                for output_name in self.output_names})
                max_lengths.update({output_name: kwargs['max_length']
                                    for output_name in self.output_names})
            else:
                # Assumes that lengths were not changed.
                lengths.update({output_name: lengths[self.input_names[0]]
                                for output_name in self.output_names})
                max_lengths.update({output_name: max_lengths[self.input_names[0]]
                                    for output_name in self.output_names})

        return output

    def get_config_as_json(self):
        return jsonpickle.encode(self.config, indent=4)

    def __getattr__(self, item):
        try:
            return super().__getattr__(item)
        except AttributeError as e:
            if item != "model":
                return getattr(self.model, item)
            else:
                raise e
