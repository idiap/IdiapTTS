#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#
from enum import Enum
from functools import partial
import logging
import os
from types import MethodType
from typing import Callable, Dict, List, Tuple, Union
from idiaptts.misc.normalisation.MeanCovarianceExtractor import MeanCovarianceExtractor
from idiaptts.misc.normalisation.MeanStdDevExtractor import MeanStdDevExtractor
from idiaptts.misc.normalisation.MinMaxExtractor import MinMaxExtractor

import numpy as np

from idiaptts.src.data_preparation.DataReaderConfig import get_length, _get_padding_sizes
from numpy.compat.py3k import npy_load_module


class DataReader:

    class Config:
        def __init__(
                self,
                name: str,
                chunk_size: int = 1,
                match_length: Union[str, List[str]] = None,
                output_names: List[str] = None,
                random_select: bool = True,
                max_frames: int = None,
                min_frames: int = None,
                pad_mode: str = 'constant',
                other_pad_dims: List[int] = None,
                requires_seq_mask: bool = False):

            self.name = name
            self.chunk_size = chunk_size

            if type(match_length) in (tuple, list) or match_length is None:
                self.match_length = match_length
            else:
                self.match_length = (match_length,)

            if output_names is None:
                self.output_names = (name,)
            else:
                if (isinstance(output_names, tuple)
                        or isinstance(output_names, list)):
                    self.output_names = output_names
                else:
                    self.output_names = (output_names,)

            self.random_select = random_select
            self.max_frames = max_frames
            self.min_frames = min_frames
            self.pad_mode = pad_mode
            self.other_pad_dims = other_pad_dims
            self.requires_seq_mask = requires_seq_mask

    def __init__(self, config: Config) -> None:
        self._length_cache = {}
        self.get_length = MethodType(partial(
            get_length, chunk_size=config.chunk_size), self)

        self.name = config.name
        self.match_length = config.match_length
        self.random_select = config.random_select
        self.max_frames = config.max_frames
        self.min_frames = config.min_frames
        self.pad_mode = config.pad_mode
        self.other_pad_dims = config.other_pad_dims
        self.chunk_size = config.chunk_size
        self.output_names = config.output_names
        self.requires_seq_mask = config.requires_seq_mask

    def __getitem__(self, id_name: str) -> Dict[str, object]:
        item = self.load(id_name)
        item = self.preprocess_sample(item)
        if type(item) in (tuple, list):
            num_items = len(item)
        else:
            num_items = 1
            item = (item,)

        num_output_names = len(self.output_names)
        if num_items != num_output_names:
            raise RuntimeError("The data reader returns {} item(s) but {} "
                               "output names were given.".format(
                                   num_items, num_output_names))

        if self.chunk_size > 1:
            item = [self.pad(i, _get_padding_sizes(i, self.chunk_size))
                    for i in item]

        output_dict = {output_name: value for output_name, value in
                       zip(self.output_names, item)}
        output_dict["_id_list"] = id_name
        return output_dict

    def load(self, id_name: str):
        raise NotImplementedError(
            "Class {} doesn't implement load(id_name).".format(
                self.__class__.__name__))

    def preprocess_sample(self, id_name: str):
        raise NotImplementedError(
            "Class {} doesn't implement preprocess_sample(id_name).".format(
                self.__class__.__name__))

    def pad(self, sample, pad_width, pad_mode=None):
        if pad_mode is None:
            pad_mode = self.pad_mode
        return np.pad(sample, pad_width, pad_mode)

    @staticmethod  # TODO: Remove and use trim instead.
    def trim_end_sample(sample, length, reverse=False):
        """
        Trim the end of a sample by the given length. If reverse is True,
        the front of the sample is trimmed. This function is called after
        preprocess_sample.
        """
        if reverse:
            trim_width = (0, length)
        else:
            trim_width = (length, 0)
        return NpzDataReader.trim(None, sample=sample, trim_width=trim_width)

    def trim(self, sample, trim_width):
        if (np.array(trim_width) == 0).all():
            return sample
        trim_width = [slice(v[0], sample.shape[dim] - v[1])
                      if isinstance(v, tuple) else v
                      for dim, v in enumerate(trim_width)]
        return sample[tuple(trim_width)]


class NpzDataReader(DataReader):
    class Config(DataReader.Config):

        class NormType(Enum):
            NONE = "None"
            MEAN_VARIANCE = "mean_variance"
            MEAN_STDDEV = "mean_stddev"
            MIN_MAX = "min_max"

        def __init__(
                self,
                name: str,
                # TODO: Change to directories everywhere.
                directory: Union[List[os.PathLike], os.PathLike] = None,
                features: Union[str, List[str]] = None,
                indices: Union[np.ndarray,
                               Dict[int, Union[slice, np.ndarray]]] = None,
                norm_params_path: os.PathLike = None,
                norm_params: Union[Tuple[np.ndarray, np.ndarray],
                                   List[Tuple[np.ndarray, np.ndarray]]] = None,
                norm_type: str = NormType.NONE,
                output_names: List[str] = None,
                preprocessing_fn: Callable = None,
                preprocess_before_norm: bool = False,
                postprocessing_fn: Callable = None,
                postprocess_before_norm: bool = True,
                **kwargs) -> None:
            """
            TODO:
            [summary]

            :param name: [description]
            :type name: str
            :param directory: [description], defaults to None
            :type directory: Union[List[os.PathLike], os.PathLike], optional
            :param features: [description], defaults to None
            :type features: Union[str, List[str]], optional
            :param indices: Numpy array to select a subset of the loaded features. By default it selects in the last
                            dimension. It can also be a dictionary of the form
                                {0: indices of first dim,
                                 2: indices of third dim},
                            missing dimensions are taken completely. Defaults to None
            :type indices: Union[np.ndarray, Dict[int, Union[slice, np.ndarray]]], optional
            :param norm_params_path: [description], defaults to None
            :type norm_params_path: os.PathLike, optional
            :param norm_params: [description], defaults to None
            :type norm_params: Union[Tuple[np.ndarray, np.ndarray], List[Tuple[np.ndarray, np.ndarray]]], optional
            :param output_names: [description], defaults to None
            :type output_names: List[str], optional
            :param preprocessing_fn: [description], defaults to None
            :type preprocessing_fn: Callable, optional
            :param preprocess_before_norm: [description], defaults to False
            :type preprocess_before_norm: Boolean, optional
            :param postprocessing_fn: [description], defaults to None
            :type postprocessing_fn: Callable, optional
            :param postprocess_before_norm: [description], defaults to True
            :type postprocess_before_norm: Boolean, optional
            """

            if features is None:
                self.features = self._str_to_list(name)
            else:
                self.features = self._str_to_list(features)

            self.indices = indices
            if indices is not None and type(indices) is not dict:
                self.indices = indices.astype(np.long)

            if output_names is None:
                self.output_names = self.features
            else:
                self.output_names = output_names

            super().__init__(name=name, output_names=self.output_names, **kwargs)

            if type(directory) not in (tuple, list):
                self.directory = [directory]
            else:
                self.directory = directory
            self.norm_params_path = norm_params_path
            self.norm_params = norm_params
            self.norm_type = norm_type
            self.preprocessing_fn = preprocessing_fn
            self.preprocess_before_norm = preprocess_before_norm
            self.postprocessing_fn = postprocessing_fn
            self.postprocess_before_norm = postprocess_before_norm

        @staticmethod
        def _str_to_list(string: Union[str, List[str]]):
            return string if type(string) is list else [string]

        def create_reader(self):
            return NpzDataReader(self)

    def __init__(self, config: Config) -> None:
        super().__init__(config)

        self.directory = config.directory
        self.features = config.features
        self.indices = config.indices

        if config.norm_type == NpzDataReader.Config.NormType.NONE:
            self.normaliser = None
        elif config.norm_type == NpzDataReader.Config.NormType.MEAN_VARIANCE:
            self.normaliser = MeanCovarianceExtractor()
        elif config.norm_type == NpzDataReader.Config.NormType.MEAN_STDDEV:
            self.normaliser = MeanStdDevExtractor()
        elif config.norm_type == NpzDataReader.Config.NormType.MIN_MAX:
            self.normaliser = MinMaxExtractor()
        else:
            raise NotImplementedError("Unknown norm_type {}".format(
                config.norm_type))

        if config.norm_params is not None:
            self.norm_params = config.norm_params
        elif config.norm_params_path is not None:
            self.norm_params = self.normaliser.load(config.norm_params_path)
        else:
            self.norm_params = None

        self.preprocessing_fn = config.preprocessing_fn
        self.preprocess_before_norm = config.preprocess_before_norm
        self.postprocessing_fn = config.postprocessing_fn
        self.postprocess_before_norm = config.postprocess_before_norm

    def get_normalisation_params(self, dir_out=None, file_name=None):
        """
        Read the normalisation parameters from files. Saves them in
        self.norm_params.

        :param dir_out:    Directory containing the normalisation file.
                           If None, self.directory is searched. If there
                           is more than one file found, self.norm_params
                           becomes a list.
        :param file_name:  Prefix of normalisation file. Expects file to
                           be named <file_name-><self.normaliser.file_name_appendix>.npz
        :return:           Tuple of normalisation parameters (depends
                           on norm_type) or list of tuples.
        """
        if self.normaliser is not None:
            if dir_out is None:
                self.norm_params = list()
                for directory in self.directory:
                    try:
                        self.norm_params.append(self._load_normalisation_params(
                            directory, file_name))
                    except FileNotFoundError:
                        pass
                assert len(self.norm_params) > 0, "No normalisation parameter"\
                    " file found in self.directory and dir_out was None."
                if len(self.norm_params) == 1:
                    self.norm_params = self.norm_params[0]
            else:
                self._load_normalisation_params(dir_out, file_name)

        return self.norm_params

    def _load_normalisation_params(self, directory, file_name=None):
        if file_name is None:
            file_name = ""
        elif os.path.basename(file_name) != "":
            file_name += "-"
        full_file_name = file_name + self.normaliser.file_name_appendix

        try:
            self.norm_params = self.normaliser.load(os.path.join(
                directory, full_file_name + ".npz"))
        except FileNotFoundError:
            # LEGACY support
            self.norm_params = self.normaliser.load(os.path.join(
                directory, full_file_name + ".bin"))

    def load(self, id_name: str) -> List[np.ndarray]:
        # Features should be stored in same directory, no speaker
        # dependent subdirectories.
        id_name = os.path.splitext(os.path.basename(id_name))[0]

        missing_features = self.features
        feature_list = []
        for dir in self.directory:
            file_path = os.path.join(dir, id_name + ".npz")
            if os.path.isfile(file_path):
                npz_file = np.load(file_path)

                found_features = []
                for feature_name in missing_features:
                    if feature_name in npz_file:
                        features = npz_file[feature_name]
                        features = features.astype(np.float32, copy=False)
                        feature_list.append(features)
                        found_features.append(feature_name)

                missing_features = [feature for feature in missing_features
                                    if feature not in found_features]

        if len(missing_features) == 0:
            if len(feature_list) == 1:  # TODO: Is this needed?
                return feature_list[0]
            else:
                return feature_list
        else:
            raise FileNotFoundError("Cannot find file {}.npz or features {} in"
                                    " it in [{}]".format(
                                        id_name,
                                        ", ".join(missing_features),
                                        ",".join(self.directory)))

    def preprocess_sample(self,
                          features: Union[Dict[str, np.ndarray], np.ndarray],
                          feature_idx: int = 0):

        if isinstance(features, list):
            preprocessed_features = list()
            for feature_idx, feature in enumerate(features):
                preprocessed_features.append(self.preprocess_sample(
                    feature, feature_idx=feature_idx))
            return preprocessed_features

        if self.indices is not None:
            features = self._get_features_subset(features)

        if self.preprocess_before_norm and self.preprocessing_fn is not None:
            features = self.preprocessing_fn(features)

        if self.normaliser is not None:
            features = self._normalise_feature(features, feature_idx)

        if not self.preprocess_before_norm \
                and self.preprocessing_fn is not None:
            features = self.preprocessing_fn(features)

        return features.astype(np.float32, copy=False)

    def _get_features_subset(self, features: np.ndarray):
        if type(self.indices) is dict:
            index_tuple = [self.indices.get(dim, slice(None))
                           for dim in range(features.ndim)]
            return features[index_tuple]
        else:
            return features[..., self.indices]

    def _normalise_feature(self, feature, feature_idx):
        assert type(feature) not in [tuple, list], \
            "Multiple features not supported."
        if self.norm_params is None:
            raise ValueError("norm_params not set, call "
                             "get_normalisation_params() before.")
        if type(self.norm_params[0]) in [tuple, list]:
            norm_params = self.norm_params[feature_idx]
        else:
            norm_params = self.norm_params
        return self.normaliser._normalise(feature, *norm_params)

    def postprocess_sample(self, features, feature_idx=0):
        if isinstance(features, dict):
            postprocessed_features = dict()
            for feature_idx, feature_name in enumerate(self.features):
                postprocessed_features[feature_name] = self.postprocess_sample(
                    features[feature_name], feature_idx=feature_idx)
            return postprocessed_features

        if self.postprocess_before_norm and self.postprocessing_fn is not None:
            features = self.postprocessing_fn(features)

        if self.norm_params is not None:
            features = self._denormalise_feature(features, feature_idx)

        if not self.postprocess_before_norm \
                and self.postprocessing_fn is not None:
            features = self.postprocessing_fn(features)

        return features#.astype(np.float32, copy=False)

    def _denormalise_feature(self, feature, feature_idx):
        assert type(feature) not in [tuple, list], \
            "Multiple features not supported."
        if type(self.norm_params[0]) in [tuple, list]:
            norm_params = self.norm_params[feature_idx]
        else:
            norm_params = self.norm_params
        return self.normaliser._denormalise(feature, *norm_params)
