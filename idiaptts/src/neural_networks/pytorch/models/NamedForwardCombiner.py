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


class NamedForwardCombiner(NamedForwardModule):

    class Config(ModelConfig):
        def __init__(self,
                     input_names: List[str],
                     batch_first: bool,
                     output_names: str,
                     input_merge_type: str = ModelConfig.MERGE_TYPE_CAT):
            super().__init__(input_names=input_names, batch_first=batch_first, input_merge_type=input_merge_type,
                             name="Combiner[{}->{}]".format(" ".join(input_names), output_names),
                             output_names=[output_names])

        def create_model(self):
            return NamedForwardCombiner(self)

    def __init__(self, config: Config):
        super().__init__(input_names=config.input_names,
                         input_merge_type=config.input_merge_type,
                         name=config.name,
                         output_names=config.output_names,
                         batch_first=config.batch_first)

    # def forward(self, data, lengths, max_lengths, **kwargs):
        # input_, length = NamedForwardModule._get_inputs(data, self.input_names, self.input_merge_type,
        #                                                  self.batch_first, return_lengths=True)
        # output = input_
        # output_dict = self._map_to_output_names(output)
        # data.update(output_dict)

        # lengths.update({name: length for name in self.output_names})
        # max_length = max(length) if len(length) > 0 else 1
        # max_lengths.update({name: max_length for name in self.output_names})

    def forward_module(self, input_, lengths, max_length, **kwargs):
        input_lengths = [lengths[name] for name in self.input_names]
        input_max_length = [max_length[name] for name in self.input_names]
        output_lengths = reduce(torch.max, input_lengths)
        output_max_length = reduce(torch.max, input_max_length)
        lengths.update({name: output_lengths for name in self.output_names})
        max_length.update({name: output_max_length for name in self.output_names})
        return input_

    def get_config_as_json(self):
        return jsonpickle.encode(self.config, indent=4)
