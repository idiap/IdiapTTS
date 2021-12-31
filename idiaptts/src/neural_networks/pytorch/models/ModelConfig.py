#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#

from typing import Union, Any, List, Optional, cast, Dict


class ModelConfig:
    MERGE_TYPE_ADD = "add"
    MERGE_TYPE_CAT = "cat"
    MERGE_TYPE_LIST = "list"
    MERGE_TYPE_MEAN = "mean"
    MERGE_TYPE_MUL = "mul"
    MERGE_TYPE_ATTENTION = "attention"

    @staticmethod
    def _get_input_dim(input_names: List[str], datareader: Dict):
        in_dim = 0
        for input_name in input_names:
            in_dim += datareader[input_name].get_dim()
        return in_dim

    @staticmethod
    def _str_to_list(str_or_list):
        if str_or_list is None:
            return None
        elif type(str_or_list) in [tuple, list]:
            return str_or_list
        else:
            return [str_or_list]

    def __init__(self,
                 input_names: List[str],
                 batch_first: bool,
                 input_merge_type: str = MERGE_TYPE_CAT,
                 name: str = None,
                 output_names: List[str] = None,
                 **kwargs):
        super().__init__()

        self.input_names = self._str_to_list(input_names)
        self.batch_first = batch_first
        self.input_merge_type = input_merge_type
        self.name = name
        assert output_names is not None or name is not None, \
            "Default output_names is [name], but both are None for input {}.".format(input_names)
        self.output_names = self._str_to_list(output_names) if output_names is not None else [name]
        self.kwargs = kwargs

    def create_model(self):
        raise NotImplementedError()
