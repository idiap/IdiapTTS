#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#

import copy
from functools import reduce
from typing import Union, Any, List, Optional, cast, Dict

import jsonpickle
import torch

from idiaptts.src.neural_networks.pytorch.models.ModelConfig import ModelConfig
from idiaptts.src.neural_networks.pytorch.models.NamedForwardModule import NamedForwardModule


class NamedForwardSplitter(NamedForwardModule):

    class Config(ModelConfig):
        def __init__(self,
                     input_names: List[str],
                     batch_first: bool,
                     output_names: List[str],
                     split_sizes: Union[int, List[int]],
                     input_merge_type: str = ModelConfig.MERGE_TYPE_CAT,
                     split_dim: int = -1):
            super().__init__(input_names=input_names, batch_first=batch_first, input_merge_type=input_merge_type,
                             name="Splitter[{}->{}]".format(" ".join(input_names), " ".join(output_names)),
                             output_names=output_names)
            self.split_sizes = split_sizes if type(split_sizes) in [list, tuple] else (split_sizes,)
            self.split_dim = split_dim

        def create_model(self):
            return NamedForwardSplitter(self)

    def __init__(self, config: Config):
        super().__init__(input_names=config.input_names,
                         input_merge_type=config.input_merge_type,
                         name=config.name,
                         output_names=config.output_names,
                         batch_first=config.batch_first)
        self.split_sizes = config.split_sizes
        self.split_dim = config.split_dim

    def forward_module(self, input_, lengths, max_length, **kwargs):
        input_lengths = [lengths[name] for name in self.input_names]
        input_max_length = [max_length[name] for name in self.input_names]
        output_lengths = reduce(torch.max, input_lengths)
        output_max_length = reduce(torch.max, input_max_length)
        lengths.update({name: output_lengths for name in self.output_names})
        max_length.update({name: output_max_length for name in self.output_names})

        return torch.split(input_, self.split_sizes, self.split_dim)

    def get_config_as_json(self):
        return jsonpickle.encode(self.config, indent=4)
