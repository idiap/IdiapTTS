#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#
import os
from typing import Union, Any, List, Optional, cast
from types import MethodType
from functools import partial


def get_length(self, id_name, chunk_size=1):
    try:
        return self._length_cache[id_name]
    except KeyError:
        reader_output = self[id_name]
        if isinstance(reader_output, dict):
            length = max(len(v) for k, v in reader_output.items() if k != "_id_list")
        else:
            length = len(reader_output)
        length = ((length + chunk_size - 1) // chunk_size) * chunk_size
        self._length_cache[id_name] = length
        return length


def _get_padding_sizes(item, chunk_size):
    length = len(item)
    padding_first_dim = (0, ((length + chunk_size - 1) // chunk_size) * chunk_size - length)
    padding_other_dims = ([(0, 0)] * (item.ndim - 1))
    return (padding_first_dim, *padding_other_dims)


class DataReaderConfig():

    # _legacy_directory_map = {
    #     "PhonemeLabelGen": "dir_labels",
    #     "PhonemeDurationLabelGen": "dir_labels"
    # }

    def __init__(self,
                 name: str,
                 feature_type,
                 directory: Union[str, os.PathLike] = None,
                 features: Union[str, List[str]] = None,
                 output_names: List[str] = None,
                 match_length: Union[str, List[str]] = None,
                 min_frames: int = None,
                 max_frames: int = None,
                 pad_mode: str = 'constant',
                 other_pad_dims: List[int] = None,
                 random_select: bool = False,
                 chunk_size: int = 1,
                 requires_seq_mask: bool = False,
                 **kwargs):
        """
        [summary]

        :param name: [description]
        :type name: str
        :param feature_type: [description]
        :type feature_type: [type]
        :param directory: [description], defaults to None
        :type directory: Union[str, os.PathLike], optional
        :param features: [description], defaults to None
        :type features: Union[str, List[str]], optional
        :param output_names: Assign explicit names to each output (needed for multiple features), defaults to name
        :type output_names: List[str], optional
        :param match_length: Name of reference data reader for each dimension, defaults to None
        :type match_length: Union[str, List[str]], optional
        :param min_frames: [description], defaults to None
        :type min_frames: int, optional
        :param max_frames: [description], defaults to None
        :type max_frames: int, optional
        :param random_select: [description], defaults to False
        :type random_select: bool, optional
        :param chunk_size: Pad time dimension to multiples of it, defaults to 1
        :type chunk_size: int, optional
        :param requires_seq_mask: Whether the output features require a sequence mask in the data dict, added by the
                              ModularModelHandler. The mask is used in the losses, defaults to False
        :type requires_seq_mask: bool, optional
        """
        self.name = name
        self.type = feature_type
        self.directory = directory

        if features is None:
            self.features = DataReaderConfig._str_to_list(name)
        else:
            self.features = DataReaderConfig._str_to_list(features)

        if output_names is None:
            self.output_names = self.features
        else:
            self.output_names = output_names

        self.match_length = match_length if type(match_length) in (tuple, list)\
            or match_length is None else (match_length,)
        self.min_frames = min_frames
        self.max_frames = max_frames
        self.pad_mode = pad_mode
        self.other_pad_dims = other_pad_dims
        self.random_select = random_select
        self.chunk_size = chunk_size
        self.requires_seq_mask = requires_seq_mask

        self.kwargs = kwargs

    @staticmethod
    def _str_to_list(string: Union[str, List[str]]):
        return string if type(string) is list else [string]

    def create_reader(self):
        # if self.type in self._legacy_directory_map:
        #     mapped_name = self._legacy_directory_map[self.type]
        #     self.kwargs[mapped_name] = self.directory

        import idiaptts.src.data_preparation as datareader_classes
        reader = getattr(datareader_classes, self.type)(dir_labels=self.directory, **self.kwargs)  # TODO: Datareaders should use self.features to load the features with that name from the npz files.
        if hasattr(reader, "get_normalisation_params") and callable(reader.get_normalisation_params):
            reader.get_normalisation_params()

        self._set_get_item_method(reader)

        # TODO: Move this into datareader baseclass
        reader._length_cache = {}
        reader.get_length = MethodType(partial(get_length, chunk_size=self.chunk_size), reader)
        # setattr(reader, 'get_length', get_length)

        # TODO: Move this into datareader baseclass
        reader.name = self.name
        reader.match_length = self.match_length
        reader.min_frames = self.min_frames
        reader.max_frames = self.max_frames
        reader.pad_mode = self.pad_mode
        reader.other_pad_dims = self.other_pad_dims
        reader.random_select = self.random_select
        reader.chunk_size = self.chunk_size
        reader.output_names = self.output_names
        reader.requires_seq_mask = self.requires_seq_mask

        return reader

    def _set_get_item_method(self, reader):
        # TODO: Add this functionality to datareader baseclass.
        # reader.__getitem__ = MethodType(lambda s, id_name: {output_name: value for output_name, value in zip(self.output_names, super(type(s), s).__getitem__(id_name))}, reader)
        reader.__getitem__ = MethodType(partial(self.get_named_item,
                                                output_names=self.output_names,
                                                get_item_fn=reader.__getitem__,
                                                chunk_size=self.chunk_size,
                                                pad_fn=reader.pad),
                                        reader)

    @staticmethod
    def get_named_item(self, id_name, output_names, get_item_fn, chunk_size=1, pad_fn=None):
        item = get_item_fn(id_name)
        if type(item) in (tuple, list):
            num_items = len(item)
        else:
            num_items = 1
            item = (item,)

        num_output_names = len(output_names)
        if num_items != num_output_names:
            raise RuntimeError("The data reader returns {} item(s) but {} output names were given."
                               .format(num_items, num_output_names))

        if chunk_size > 1:
            item = [pad_fn(i, _get_padding_sizes(i, chunk_size)) for i in item]

        output_dict = {output_name: value for output_name, value in zip(output_names, item)}
        output_dict["_id_list"] = id_name
        return output_dict
