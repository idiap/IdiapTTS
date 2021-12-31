#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#

"""Module description:
   Create world feature labels for .wav files.
"""


# System imports.
import argparse
from collections import OrderedDict
from functools import partial
import glob
import logging
import math
import os
import sys
from typing import Callable, Dict, List, Union, Tuple
from idiaptts.src.data_preparation.NpzDataReader import NpzDataReader
from idiaptts.src.data_preparation.audio.AudioProcessing import AudioProcessing

# Third-party imports.
import numpy as np
import pyworld
import scipy

# Local source tree imports.
from idiaptts.misc.mlpg import MLPG
from idiaptts.misc.normalisation.MeanCovarianceExtractor import MeanCovarianceExtractor
from idiaptts.misc.normalisation.MeanStdDevExtractor import MeanStdDevExtractor
from idiaptts.misc.utils import makedirs_safe, interpolate_lin, compute_deltas
from idiaptts.src.data_preparation.audio.AudioProcessing import AudioProcessing
from idiaptts.src.data_preparation.LabelGen import LabelGen


class WorldFeatLabelGen(NpzDataReader, LabelGen):
    """Create world feat labels for .wav files."""

    # TODO: Constants into hparams?
    f0_silence_threshold = 30
    lf0_zero = 0
    preemphasis = 0.0
    n_fft = None
    win_length_ms = None

    dir_lf0 = "lf0"
    dir_vuv = "vuv"
    dir_bap = "bap"
    dir_deltas = "cmp"

    ext_lf0 = "lf0"
    ext_vuv = "vuv"
    ext_bap = "bap"
    ext_deltas = "cmp"

    logger = logging.getLogger(__name__)

    class Config(NpzDataReader.Config):
        def __init__(
                self,
                name: str,
                directory: Union[List[os.PathLike], os.PathLike] = None,
                # features: Union[str, List[str]] = None,
                indices: Union[np.ndarray,
                               Dict[int, Union[slice, np.ndarray]]] = None,
                norm_params_path: os.PathLike = None,
                norm_params: Union[Tuple[np.ndarray, np.ndarray],
                                   List[Tuple[np.ndarray, np.ndarray]]] = None,
                norm_type: str = NpzDataReader.Config.NormType.MEAN_VARIANCE,
                output_names: List[str] = None,
                preprocessing_fn: Callable = None,
                preprocess_before_norm: bool = False,
                postprocessing_fn: Callable = None,
                postprocess_before_norm: bool = False,
                add_deltas: bool = False,
                preemphasis: float = 0.0,
                n_fft: int = None,
                win_length_ms: int = None,
                num_coded_sps: int = 60,
                num_bap: int = 1,
                sp_type: str = "mcep",
                load_sp: bool = True,
                load_lf0: bool = True,
                load_vuv: bool = True,
                load_bap: bool = True,
                apply_mlpg: bool = True,
                **kwargs) -> None:

            if postprocessing_fn is None:
                postprocessing_fn=partial(WorldFeatLabelGen._postprocess_world,
                                          apply_mlpg=apply_mlpg),

            if norm_type is None:  # TODO: This might be missleading.
                if add_deltas:
                    norm_type = NpzDataReader.Config.NormType.MEAN_VARIANCE
                else:
                    norm_type = NpzDataReader.Config.NormType.MEAN_STDDEV

            super().__init__(name,
                             directory=directory,
                             indices=indices,
                             norm_params_path=norm_params_path,
                             norm_params=norm_params,
                             norm_type=norm_type,
                             output_names=output_names,
                             preprocessing_fn=preprocessing_fn,
                             preprocess_before_norm=preprocess_before_norm,
                             postprocessing_fn=postprocessing_fn,
                             postprocess_before_norm=postprocess_before_norm,
                             **kwargs)

            if type(directory) in [tuple, list]:
                self.dir_labels = directory[0]
            else:
                self.dir_labels = directory
            self.add_deltas = add_deltas
            self.preprocessing_fn = preprocessing_fn
            self.preemphasis = preemphasis
            self.n_fft = n_fft
            self.win_length_ms = win_length_ms
            self.num_coded_sps = num_coded_sps
            self.num_bap = num_bap
            self.sp_type = sp_type
            self.load_sp = load_sp
            self.load_lf0 = load_lf0
            self.load_vuv = load_vuv
            self.load_bap = load_bap

        def create_reader(self):
            reader = WorldFeatLabelGen(self)

            reader.get_normalisation_params()

            return reader

    def __init__(self, *args, **kwargs):
        """
        Constructor to use the class as a database. If add_delta is
        false labels have the dimension num_frames x (num_coded_sps + 3)
        [sp_type(num_coded_sps), lf0, vuv, bap(1)], otherwise the deltas
        and double deltas are added between the features resulting in
        num_frames x (3*num_coded_sps + 7) [sp_type(3*num_coded_sps),
        lf0(3*1), vuv, bap(3*1)].

        :param dir_labels:     While using it as a database dir_labels
                               has to contain the prepared labels.
        :param add_deltas:     Determines if labels contain deltas and
                               double deltas.
        :param sampling_fn:    Provide a function for up- or down-
                               sampling the features during preprocessing.
        :param num_coded_sps:  Number of bins used to represent the
                               coded spectral features.
        :param sp_type:        Type of the encoded spectral features
                               e.g. mcep, mgc, mfbanks (mel-filter banks).
        :param hop_size_ms:    Hop size of FFT window in milliseconds.
        :param load_sp:        Whether to extract/load coded spectral
                               features.
        :param load_lf0:       Whether to extract/load LF0.
        :param load_vuv:       Whether to extract/load V/UV flag.
        :param load_bap:       Whether to extract/load BAP.
        """
        if len(args) == 1 and isinstance(args[0], WorldFeatLabelGen.Config):
            config = args[0]

            super().__init__(config)

            self.dir_labels = config.dir_labels
            self.add_deltas = config.add_deltas
            self.preprocessing_fn = config.preprocessing_fn
            self.preemphasis = config.preemphasis
            self.n_fft = config.n_fft
            self.win_length_ms = config.win_length_ms
            self.num_coded_sps = config.num_coded_sps
            self.num_bap = config.num_bap
            self.sp_type = config.sp_type
            self.load_sp = config.load_sp
            self.load_lf0 = config.load_lf0
            self.load_vuv = config.load_vuv
            self.load_bap = config.load_bap

            self.legacy_getitem = False
        else:
            # LEGACY support
            if "dir_labels" in kwargs:
                self.dir_labels = kwargs["dir_labels"]
            else:
                self.dir_labels = args[0]

            default_kwargs = {"add_deltas": False,
                              "preprocessing_fn": None,
                              "preemphasis": 0.0,
                              "n_fft": None,
                              "win_length_ms": None,
                              "num_coded_sps": 60,
                              "num_bap": 1,
                              "sp_type": "mcep",
                              "hop_size_ms": 5,
                              "load_sp": True,
                              "load_lf0": True,
                              "load_vuv": True,
                              "load_bap": True}
            default_kwargs.update(kwargs)
            self.add_deltas = default_kwargs["add_deltas"]
            self.preemphasis = default_kwargs["preemphasis"]
            self.n_fft = default_kwargs["n_fft"]
            self.win_length_ms = default_kwargs["win_length_ms"]
            self.num_coded_sps = default_kwargs["num_coded_sps"]
            self.num_bap = default_kwargs["num_bap"]
            self.sp_type = default_kwargs["sp_type"]
            self.load_sp = default_kwargs["load_sp"]
            self.load_lf0 = default_kwargs["load_lf0"]
            self.load_vuv = default_kwargs["load_vuv"]
            self.load_bap = default_kwargs["load_bap"]

            super().__init__(WorldFeatLabelGen._get_npz_reader_config(
                dir_labels=self.dir_labels,
                preprocessing_fn=default_kwargs["preprocessing_fn"]
            ))

            self.hop_size_ms = default_kwargs["hop_size_ms"]

            self.legacy_getitem = True

        self.norm_params = None
        # Leave space for V/UV covariance even though it is never used.
        self.covs = [None] * 4
        self.dir_coded_sps = self.sp_type
        if self.num_coded_sps != -1:
            self.dir_coded_sps += str(self.num_coded_sps)
        self.dir_deltas += "_" + self.dir_coded_sps

        def get_features(feature_name):
            if self.add_deltas and feature_name != WorldFeatLabelGen.ext_vuv:
                return [feature_name, feature_name + "_deltas",
                        feature_name + "_double_deltas"]
            else:
                return feature_name

        if self.add_deltas:
            norm_type = NpzDataReader.Config.NormType.MEAN_VARIANCE
        else:
            norm_type = NpzDataReader.Config.NormType.MEAN_STDDEV

        datareader_sp = NpzDataReader.Config(
            name="acoustic_features",
            directory=os.path.join(self.dir_labels,
                                   self.sp_type + str(self.num_coded_sps)),
            features=get_features(self.sp_type),
            norm_type=norm_type
        ).create_reader() if self.load_sp else None

        datareader_lf0 = NpzDataReader.Config(
            name="acoustic_features",
            directory=os.path.join(self.dir_labels, WorldFeatLabelGen.dir_lf0),
            features=get_features(WorldFeatLabelGen.ext_lf0),
            norm_type=norm_type
        ).create_reader() if self.load_lf0 else None

        datareader_vuv = NpzDataReader.Config(
            name="acoustic_features",
            directory=os.path.join(self.dir_labels, WorldFeatLabelGen.dir_vuv),
            features=WorldFeatLabelGen.ext_vuv,
            norm_type=norm_type
        ).create_reader() if self.load_vuv else None

        datareader_bap = NpzDataReader.Config(
            name="acoustic_features",
            directory=os.path.join(self.dir_labels, WorldFeatLabelGen.dir_bap),
            features=get_features(WorldFeatLabelGen.ext_bap),
            norm_type=norm_type
        ).create_reader() if self.load_bap else None

        self.load_flags = (self.load_sp, self.load_lf0, self.load_vuv,
                           self.load_bap)
        self.datareaders = (datareader_sp, datareader_lf0, datareader_vuv,
                            datareader_bap)

    @staticmethod
    def _get_npz_reader_config(dir_labels, preprocessing_fn, apply_mlpg=False):
        return NpzDataReader.Config(
            name="acoustic_features",
            directory=dir_labels,
            norm_type=NpzDataReader.Config.NormType.MEAN_VARIANCE,
            preprocessing_fn=preprocessing_fn,
            preprocess_before_norm=False,
            postprocessing_fn=partial(WorldFeatLabelGen._postprocess_world,
                                      apply_mlpg=apply_mlpg),
            postprocess_before_norm=False)

    def __getitem__(self, id_name):
        """Return the preprocessed sample with the given id_name."""

        sample_dict = super().__getitem__(id_name)

        if self.legacy_getitem:
            # LEGACY support
            return sample_dict[self.output_names[0]]
        else:
            return sample_dict

    def _get_norm_params_subset(self, norm_params):
        mean, std_dev = norm_params

        processed_params = 0
        num_sp_features = self.num_coded_sps * (3 if self.add_deltas else 1)
        if not self.load_sp:
            mean = mean[:num_sp_features]
            std_dev = std_dev[:num_sp_features]
        else:
            processed_params += num_sp_features

        num_lf0 = 3 if self.add_deltas else 1
        if not self.load_lf0:
            mean = mean[processed_params:processed_params + num_lf0]
            std_dev = std_dev[processed_params:processed_params + num_lf0]
        else:
            processed_params += num_lf0

        if not self.load_vuv:
            mean = mean[processed_params:processed_params + 1]
            std_dev = std_dev[processed_params:processed_params + 1]
        else:
            processed_params += 1

        num_bap = self.num_bap * (3 if self.add_deltas else 1)
        if not self.load_bap:
            mean = mean[processed_params:processed_params + num_bap]
            std_dev = std_dev[processed_params:processed_params + num_bap]
        else:
            processed_params += num_bap

        return mean, std_dev

    def postprocess_sample(self, sample, norm_params=None, apply_mlpg=True):
        # LEGACY support to allow providing apply_mlpg flag in the call.

        tmp_postprocessing_fn = self.postprocessing_fn
        tmp_norm_params = self.norm_params

        self.postprocessing_fn = lambda sample: self._postprocess_world(
            sample, apply_mlpg=apply_mlpg)

        if norm_params is not None:
            self.norm_params = norm_params

        sample = super().postprocess_sample(sample)

        self.postprocessing_fn = tmp_postprocessing_fn
        self.norm_params = tmp_norm_params

        return sample

    def _postprocess_world(self, sample, norm_params=None, apply_mlpg=True):
        """
        Postprocess one sample. This function is used after inference of
        a network.

        :param sample:       The sample to post-process.
        :param norm_params:  Use this normalisation parameters instead
                             of self.norm_params.
        :param apply_mlpg:   Apply the MLPG algorithm on the
                             post-processed sample.
        :return:             Post-processed sample.
        """

        if self.add_deltas:
            output_list = list()
            mlpg = MLPG()

            num_processed_feats = 0
            if self.load_sp:
                coded_sp_full = sample[:, :self.num_coded_sps * 3]
                num_processed_feats += self.num_coded_sps * 3
                if apply_mlpg:
                    coded_sp = mlpg.generation(coded_sp_full, self.covs[0],
                                               self.covs[0].shape[0] // 3)
                else:
                    coded_sp = coded_sp_full[:, :self.num_coded_sps]
                output_list.append(coded_sp)

            if self.load_lf0:
                lf0_full = sample[:, num_processed_feats:num_processed_feats + 3]
                num_processed_feats += 3
                if apply_mlpg:
                    lf0 = mlpg.generation(lf0_full, self.covs[1],
                                          self.covs[1].shape[0] // 3)
                else:
                    lf0 = lf0_full[:, 0:1]
                output_list.append(lf0)

            if self.load_vuv:
                vuv = sample[:, num_processed_feats]
                num_processed_feats += 1
                vuv[vuv <= 0.5] = 0.0
                vuv[vuv > 0.5] = 1.0
                vuv = vuv[:, None]
                output_list.append(vuv)

            if self.load_bap:
                bap_full = sample[:, -self.num_bap * 3:]

                if apply_mlpg:
                    bap = mlpg.generation(bap_full, self.covs[3],
                                          self.covs[3].shape[0] // 3)
                else:
                    bap = bap_full[:, 0:self.num_bap]
                output_list.append(bap)

            sample = np.concatenate(output_list, axis=1)

        return sample

    @staticmethod
    def load_sample(id_name, dir_out, add_deltas=False, num_coded_sps=60,
                    num_bap=1, sp_type="mcep", load_sp=True, load_lf0=True,
                    load_vuv=True, load_bap=True):
        """
        Load world features from dir_out. It does not pre-process, use
        __getitem__ method instead.

        :param id_name:        Id of the sample.
        :param dir_out:        Directory containing the sample.
        :param add_deltas:     Determines if deltas and double deltas
                               are expected.
        :param num_coded_sps:  Number of bins used to represent the
                               coded spectral features.
        :param num_bap:        Number of bins used to represent the
                               coded band aperiodicity.
        :param sp_type:        Type in which the coded spectral features
                               are saved.
        :param load_sp:        Load spectral features defined by sp_type.
        :param load_lf0:       Load fundamental frequency.
        :param load_vuv:       Load voiced/unvoiced flag.
        :param load_bap:       Load band aperiodicity features.
        :return:               Numpy array with dimensions num_frames x
                               len(coded_sp, lf0, vuv, bap).
        """
        assert dir_out is not None, "dir_out cannot be None"

        id_name = os.path.splitext(os.path.basename(id_name))[0]
        logging.debug("Load WORLD {} features for {}.".format(
            "deltas " if add_deltas else "", id_name))

        datareader = WorldFeatLabelGen(dir_labels=dir_out,
                                       add_deltas=add_deltas,
                                       num_coded_sps=num_coded_sps,
                                       num_bap=num_bap,
                                       sp_type=sp_type,
                                       load_sp=load_sp,
                                       load_lf0=load_lf0,
                                       load_vuv=load_vuv,
                                       load_bap=load_bap)
        return datareader.load(id_name)

    def load(self, id_name: str):

        # These dims are only used in LEGACY code.
        deltas_factor = 3 if self.add_deltas else 1
        dim_coded_sp = self.num_coded_sps * deltas_factor
        dim_lf0 = 1 * deltas_factor
        dim_vuv = 1
        dim_bap = self.num_bap * deltas_factor

        try:
            output_list = list()
            for load, datareader in zip(self.load_flags, self.datareaders):
                if not load:
                    continue

                features = datareader.load(id_name)

                if (isinstance(features, tuple) or isinstance(features, list)):
                    assert self.add_deltas, "Multiple features should only be loaded " \
                        "with add_deltas=true."
                    features = np.concatenate(features, axis=1)

                output_list.append(features)


        # try:
        #     feature_sub_dir = (sp_type + str(num_coded_sps),
        #                        WorldFeatLabelGen.dir_lf0,
        #                        WorldFeatLabelGen.dir_vuv,
        #                        WorldFeatLabelGen.dir_bap)

        #     feature_extensions = (sp_type, WorldFeatLabelGen.ext_lf0,
        #                           WorldFeatLabelGen.ext_vuv,
        #                           WorldFeatLabelGen.ext_bap)

        #     feature_dims = (dim_coded_sp, dim_lf0, dim_vuv, dim_bap)

        #     output_list = list()

        #     for load, feature_dir, feature_ext, feature_dim in zip(
        #             load_flags, feature_sub_dir, feature_extensions,
        #             feature_dims):
        #         if not load:
        #             continue

        #         path = os.path.join(dir_labels, feature_dir, id_name)

        #         try:
        #             archive = np.load(path + ".npz")
        #             labels = archive[feature_ext]
        #             if add_deltas and feature_ext != WorldFeatLabelGen.ext_vuv:
        #                 deltas = archive[feature_ext + "_deltas"]
        #                 double_deltas = archive[feature_ext + "_double_deltas"]
        #                 labels = np.concatenate((labels, deltas,
        #                                          double_deltas), axis=1)
        #             output_list.append(labels)
        #         except FileNotFoundError:
        #             # LEGACY support
        #             if add_deltas:
        #                 raise
        #             else:
        #                 path += ".{}".format(feature_ext)
        #                 with open(path, 'rb') as f:
        #                     try:
        #                         feature = np.fromfile(f, dtype=np.float32)
        #                         labels = np.reshape(feature, [-1, feature_dim])
        #                     except ValueError as e:
        #                         logging.error("Cannot load labels from {}.".format(
        #                             path))
        #                         raise e
        #                 output_list.append(labels)

        except FileNotFoundError as e1:
            # Try to load from cmp folder.
            output_list = list()
            path = os.path.join(
                self.dir_labels,
                "{}_{}{}".format(WorldFeatLabelGen.dir_deltas, self.sp_type,
                                 self.num_coded_sps),
                "{}.{}".format(id_name, WorldFeatLabelGen.ext_deltas))
            with open(path, 'rb') as f:
                try:
                    cmp = np.fromfile(f, dtype=np.float32)
                    # cmp files always contain deltas.
                    total_dims = 3 * (self.num_coded_sps + 1 + self.num_bap) + dim_vuv
                    labels = np.reshape(cmp, [-1, total_dims])
                except ValueError as e:
                    logging.error("Cannot load labels from {}.".format(path))
                    raise e

            if self.load_sp:
                output_list.append(labels[:, :dim_coded_sp])

            if self.load_lf0:
                lf0_start_idx = 3 * self.num_coded_sps
                output_list.append(labels[:, lf0_start_idx:lf0_start_idx
                                          + dim_lf0])

            if self.load_vuv:
                vuv_start_idx = -3*self.num_bap-dim_vuv
                output_list.append(labels[:, vuv_start_idx:-3*self.num_bap])

            if self.load_bap:
                if dim_bap == 3 * self.num_bap:
                    output_list.append(labels[:, -3*self.num_bap:])
                else:
                    bap_start_idx = -3 * self.num_bap
                    output_list.append(labels[:, bap_start_idx:bap_start_idx
                                              + dim_bap])

        assert len(output_list) > 0, "At least one type of acoustic " \
            "feature has to be loaded."
        labels = np.concatenate(output_list, axis=1)

        return labels

    def get_normalisation_params(self, dir_out=None, file_name=None):
        """
        Read the mean std_dev values from a file. Save them in
        self.norm_params.

        :param dir_out:    Directory containing the normalisation file.
        :param file_name:  Prefix of normalisation file (underscore "_"
                           is expected as separator).
        :return:           Tuple of normalisation parameters (mean,
                           std_dev).
        """
        if dir_out is None:
            dir_out = self.dir_labels
        feature_sub_dirs = (self.dir_coded_sps, self.dir_lf0, self.dir_vuv,
                            self.dir_bap)

        try:
            all_mean = list()
            all_std_dev = list()
            for idx, (load, datareader, subdir) in enumerate(zip(
                    self.load_flags, self.datareaders, feature_sub_dirs)):

                if not load:
                    continue

                if dir_out is not None:
                    feature_dir_out = os.path.join(dir_out, subdir)
                else:
                    feature_dir_out = None

                if self.add_deltas:
                    if file_name is None or os.path.basename(file_name) == "":
                        full_file_name = "deltas"
                    else:
                        full_file_name += "-deltas"
                else:
                    full_file_name = file_name

                if subdir == self.dir_vuv:
                    all_mean.append(np.atleast_2d(0.0))
                    all_std_dev.append(np.atleast_2d(1.0))
                else:
                    norm_params = datareader.get_normalisation_params(
                        feature_dir_out, full_file_name)
                    if self.add_deltas:
                        mean, cov, std_dev = norm_params
                        self.covs[idx] = cov
                    else:
                        mean, std_dev = norm_params

                    all_mean.append(np.atleast_2d(mean))
                    all_std_dev.append(np.atleast_2d(std_dev))

            self.norm_params = np.concatenate(all_mean, axis=1),\
                np.concatenate(all_std_dev, axis=1)

            return self.norm_params

            all_mean = list()
            all_std_dev = list()

            for idx, (load, feature_dir) in enumerate(zip(self.load_flags,
                                                          feature_sub_dirs)):
                if not load:
                    continue

                full_path = os.path.join(dir_out, feature_dir, file_name)

                if feature_dir == self.dir_vuv:
                    all_mean.append(np.atleast_2d(0.0))
                    all_std_dev.append(np.atleast_2d(1.0))
                else:
                    if self.add_deltas:
                        full_path += "deltas-" \
                            + MeanCovarianceExtractor.file_name_appendix
                        mean, cov, std_dev = MeanCovarianceExtractor.load(
                            full_path + ".npz")
                        self.covs[idx] = cov
                    else:
                        full_path += MeanStdDevExtractor.file_name_appendix
                        mean, std_dev = MeanStdDevExtractor.load(
                            full_path + ".npz")

                    all_mean.append(np.atleast_2d(mean))
                    all_std_dev.append(np.atleast_2d(std_dev))

            self.norm_params = np.concatenate(all_mean, axis=1),\
                np.concatenate(all_std_dev, axis=1)

            return self.norm_params
        except FileNotFoundError as e0:
            # LEGACY support
            has_name = False
            if file_name is None:
                file_name = ""
            elif os.path.basename(file_name) != "":
                has_name = True

            if has_name:
                file_name += "-"

            output_means = list()
            output_std_devs = list()
            for load, cov_idx, feature_dir in zip(self.load_flags, range(4),
                                                  feature_sub_dirs):

                if not load:
                    continue

                if feature_dir != self.dir_vuv:
                    try:
                        path_new_style = os.path.join(
                            dir_out, self.dir_deltas,
                            "{}{}-{}.bin".format(
                                file_name,
                                feature_dir,
                                MeanCovarianceExtractor.file_name_appendix))
                        mean, cov, std_dev = MeanCovarianceExtractor.load(
                            path_new_style)
                    except FileNotFoundError as e1:
                        try:
                            # TODO: Remove legacy style path.
                            path = os.path.join(
                                dir_out, self.dir_deltas,
                                "{}{}_{}.bin".format(
                                    file_name,
                                    MeanCovarianceExtractor.file_name_appendix,
                                    feature_dir))
                            mean, cov, std_dev = MeanCovarianceExtractor.load(path)
                            self.logger.warning("Found legacy style normalisation "
                                                "parameters at {}. Consider "
                                                "recreating features or renaming "
                                                "to {}".format(path, path_new_style),
                                                DeprecationWarning)
                        except FileNotFoundError as e2:
                            raise FileNotFoundError([e0, e1, e2])

                    if not self.add_deltas:
                        no_deltas_feature_len = len(cov) // 3
                        assert len(cov) / 3 == no_deltas_feature_len,\
                            "Feature size {} is not dividable by 3. Are deltas features contained?"
                        cov = cov[:no_deltas_feature_len, :no_deltas_feature_len]
                        mean = mean[:no_deltas_feature_len]
                        std_dev = std_dev[:no_deltas_feature_len]
                    self.covs[cov_idx] = cov
                    output_means.append(np.atleast_2d(mean))
                    output_std_devs.append(np.atleast_2d(std_dev))
                else:
                    output_means.append(np.atleast_2d(0.0))
                    output_std_devs.append(np.atleast_2d(1.0))

            self.norm_params = (np.concatenate(output_means, axis=1),
                                np.concatenate(output_std_devs, axis=1))
            if self.add_deltas:
                self.norm_params = (self.norm_params[0][0],
                                    self.norm_params[1][0])

            return self.norm_params

    @staticmethod
    def convert_to_world_features(sample, contains_deltas=False,
                                  num_coded_sps=60, num_bap=1):
        """
        Convert world acoustic features w/ or w/o deltas to WORLD
        understandable features.
        """

        deltas_factor = 3 if contains_deltas else 1
        num_expected_feats = (num_coded_sps + 1 + num_bap) * deltas_factor + 1
        if sample.shape[1] != num_expected_feats:
            # Automatically detect deltas features.
            num_expected_feats = (num_coded_sps + 1 + num_bap) * 3 + 1
            if sample.shape[1] == num_expected_feats:
                deltas_factor = 3
            else:
                raise ValueError("WORLD requires all features to be present.")

        coded_sp = sample[:, :num_coded_sps]
        lf0 = sample[:, num_coded_sps * deltas_factor]
        vuv = np.copy(sample[:, num_coded_sps * deltas_factor + deltas_factor])
        vuv[vuv < 0.5] = 0.0
        vuv[vuv >= 0.5] = 1.0
        if contains_deltas:
            bap = sample[:, -num_bap * 3:-num_bap * 2]
        else:
            bap = sample[:, -num_bap:]

        return coded_sp, lf0, vuv, bap

    @staticmethod
    def convert_from_world_features(coded_sp, lf0, vuv, bap):
        """
        Convert from world features to a single feature vector with
        T x (|coded_sp|, |lf0|, |vuv|, |bap|) dim.
        """
        if lf0.ndim < 2:
            lf0 = lf0[:, None]
        if vuv.ndim < 2:
            vuv = vuv[:, None]
        if bap.ndim < 2:
            bap = bap[:, None]
        return np.concatenate((coded_sp, lf0, vuv, bap), axis=1)

    @staticmethod
    def world_extract_features(raw: np.array,
                               fs: int,
                               hop_size_ms: int,
                               f0_silence_threshold: int = None,
                               lf0_zero: float = None,
                               n_fft: int = None):
        """Extract WORLD features """
        if f0_silence_threshold is None:
            f0_silence_threshold = WorldFeatLabelGen.f0_silence_threshold
        if lf0_zero is None:
            lf0_zero = WorldFeatLabelGen.lf0_zero

        # Gives power spectrum in [0, 1]
        f0, pow_sp, ap = pyworld.wav2world(raw, fs, fft_size=n_fft,
                                           frame_period=hop_size_ms)

        amp_sp = np.sqrt(pow_sp)

        # Compute lf0 and vuv information.
        lf0 = np.log(f0.clip(min=1E-10), dtype=np.float32)
        lf0[lf0 <= math.log(f0_silence_threshold)] = lf0_zero
        lf0, vuv = interpolate_lin(lf0)
        lf0 = lf0.astype(dtype=np.float32)
        vuv = vuv.astype(dtype=np.float32)

        # Decode aperiodicity to one band aperiodicity.
        bap = np.array(pyworld.code_aperiodicity(ap, fs), dtype=np.float32)

        return amp_sp, lf0, vuv, bap

    @staticmethod
    def extract_features(dir_in, file_name: str, file_ext: str = "wav",
                         preemphasis: float = 0.0, n_fft: int = None,
                         win_length_ms: int = None, hop_size_ms: int = 5,
                         sp_type: str = "mcep",
                         num_coded_sps: int = 40, load_sp: bool = True,
                         load_lf0: bool = True, load_vuv: bool = True,
                         load_bap: bool = True,
                         f0_silence_threshold: int = None,
                         lf0_zero: float = None):
        """
        Extract acoustic features from a single audio file.
        This function is called from the gen_data function.
        """

        audio_name = os.path.join(dir_in, file_name + "." + file_ext)
        raw, fs = AudioProcessing.get_raw(audio_name, preemphasis)

        extr_features_msg = ""
        lf0, vuv, bap = None, None, None
        if sp_type in ["mcep", "mgc"] or load_lf0 or load_vuv or load_bap:
            amp_sp, lf0, vuv, bap = WorldFeatLabelGen.world_extract_features(
                raw, fs, hop_size_ms, f0_silence_threshold, lf0_zero, n_fft)

            if load_lf0:
                extr_features_msg += " WORLD lf0,"

            if load_vuv:
                # Throw a warning when less then 5% of all frames are unvoiced.
                unvoiced_frames_percentage = vuv.sum() / len(vuv) * 100.0
                if unvoiced_frames_percentage < 5.0:
                    logging.warning("Detected only {:.0f}% [{}/{}] unvoiced "
                                    "frames in {}.".format(
                                        unvoiced_frames_percentage,
                                        int(vuv.sum()),
                                        len(vuv),
                                        file_name))
                extr_features_msg += " WORLD vuv,"

            if load_bap:
                extr_features_msg += " WORLD {}bap,".format(bap.shape[1])

        coded_sp = None
        if load_sp:
            if sp_type == "mcep":
                coded_sp = AudioProcessing.extract_mcep(
                    amp_sp,
                    num_coded_sps=num_coded_sps,
                    mgc_alpha=AudioProcessing.fs_to_mgc_alpha(fs))
                assert len(coded_sp) == len(lf0), \
                    "Requires testing. Possibly trimming is a solution."
                extractor = "WORLD"
            elif sp_type == "mgc":
                coded_sp = AudioProcessing.extract_mgc(
                    amp_sp, fs=fs, num_coded_sps=num_coded_sps)
                extractor = "WORLD"
            elif sp_type in ["mfbanks", "amp_sp", "log_amp_sp"]:
                assert sp_type == "mfbanks" or num_coded_sps == -1,\
                    "Use num_coded_sps=-1 for the amplitude spectrum."
                if n_fft is None:
                    n_fft = AudioProcessing.fs_to_frame_length(fs)
                coded_sp = AudioProcessing.extract_mfbanks(
                    raw, fs=fs, n_fft=n_fft, hop_size_ms=hop_size_ms,
                    win_length_ms=win_length_ms, num_coded_sps=num_coded_sps)
                if sp_type == "log_amp_sp":
                    coded_sp = AudioProcessing.amp_to_db(coded_sp)
                extractor = "Librosa"
            extr_features_msg = "{} {}{},{}".format(extractor,
                                                    coded_sp.shape[1],
                                                    sp_type,
                                                    extr_features_msg)

        # Log some debug information.
        file_name = os.path.basename(file_name)  # Remove speaker.
        logging.info("Extracted ({}) features from {} at {} Hz with {} ms"
                     " frame hop.".format(extr_features_msg.strip().strip(','),
                                          file_name, fs, hop_size_ms))

        coded_sp, lf0, vuv, bap = WorldFeatLabelGen.trim_to_shortest(
            [coded_sp, lf0, vuv, bap])
        return coded_sp, lf0, vuv, bap

    @staticmethod
    def trim_to_shortest(features):
        len_shortest = min(map(len, [f for f in features if f is not None]))
        for idx, feature in enumerate(features):
            if feature is None:
                continue

            len_feature = len(feature)
            len_diff = len_feature - len_shortest

            if len_diff > 0:
                trim_front = len_diff // 2
                trim_end = len_diff - trim_front
                features[idx] = WorldFeatLabelGen.trim(
                    None, feature, ((trim_front, trim_end),))

        return features

    @staticmethod
    def world_features_to_raw(amp_sp: np.array, lf0: np.array, vuv: np.array,
                              bap: np.array, fs: int, n_fft: int = None,
                              f0_silence_threshold: int = None,
                              lf0_zero: float = None,
                              preemphasis: float = 0.0):
        """
        Using the world vocoder to generate the waveform. Preemphasis
        causes artifacts here.
        """
        if f0_silence_threshold is None:
            f0_silence_threshold = WorldFeatLabelGen.f0_silence_threshold
        if lf0_zero is None:
            lf0_zero = WorldFeatLabelGen.lf0_zero
        if n_fft is None:
            n_fft = AudioProcessing.fs_to_frame_length(fs)

        pow_sp = np.square(amp_sp, dtype=np.float64)

        f0 = np.exp(lf0, dtype=np.float64)
        # WORLD throws an error for too small f0 values.
        vuv[f0 < f0_silence_threshold] = 0
        f0[vuv == 0] = lf0_zero
        if f0.ndim > 1:
            assert f0.shape[1:] == (1,) * (f0.ndim - 1), \
                "F0 should have only one dimension at this stage."
            f0 = f0.squeeze()

        if bap.ndim < 2:
            bap = bap.reshape(-1, 1)

        ap = pyworld.decode_aperiodicity(np.ascontiguousarray(bap, np.float64),
                                         fs, n_fft)

        raw = pyworld.synthesize(f0, pow_sp, ap, fs).astype(np.float32,
                                                            copy=False)
        return AudioProcessing.depreemphasis(raw, preemphasis)

    def gen_data(self, dir_in: Union[str, os.PathLike],
                 dir_out: Union[str, os.PathLike] = None,
                 file_id_list: Union[str, os.PathLike] = "",
                 file_ext: str = "wav", id_list: List = None,
                 return_dict: bool = False):
        """
        Prepare acoustic features from audio files. Which features are
        extracted are determined by the parameters given in the
        constructor. The self.load_* flags determine if that features is
        extracted and the self.sp_type determines the type of coded
        spectrum representation.

        :param dir_in:        Directory where the audio files are stored
                              for each utterance to process.
        :param dir_out:       Main directory where the labels and
                              normalisation parameters are saved to
                              subdirectories. If None, labels are not
                              saved.
        :param file_id_list:  Name of the file containing the ids.
                              Normalisation parameters are saved using
                              this name to differentiate parameters
                              between subsets.
        :param file_ext:      Extension of all audio files.
        :param id_list:       The list of utterances to process. Should
                              have the form uttId1 \\n uttId2 \\n ...\\n
                              uttIdN. If None, all file in audio_dir are
                              used.
        :param return_dict:   If true, returns an OrderedDict of all
                              samples as first output_return_dict.
        :return:              Returns two normalisation parameters as
                              tuple. If return_dict is True it returns
                              all processed labels in an OrderedDict
                              followed by the two normalisation parameters.
        """
        id_list, file_id_list_name = self._get_id_list(dir_in, file_id_list,
                                                       id_list, file_ext)

        if dir_out is not None:
            self._create_directories(dir_out)

        if return_dict:
            label_dict = OrderedDict()

        self._create_norm_params_extractors()

        logging.info("Extract acoustic features{} for ".format(
            "" if not self.add_deltas else " with deltas")
                     + "[{0}]".format(", ".join(str(i) for i in id_list)))

        for file_name in id_list:
            coded_sp, lf0, vuv, bap = self.extract_features(
                dir_in, file_name, file_ext,
                preemphasis=self.preemphasis,
                n_fft=self.n_fft,
                win_length_ms=self.win_length_ms,
                sp_type=self.sp_type,
                num_coded_sps=self.num_coded_sps,
                load_sp=self.load_sp,
                load_lf0=self.load_lf0,
                load_vuv=self.load_vuv,
                load_bap=self.load_bap,
                hop_size_ms=self.hop_size_ms,
                f0_silence_threshold=WorldFeatLabelGen.f0_silence_threshold,
                lf0_zero=WorldFeatLabelGen.lf0_zero)

            output = self.save_output((coded_sp, lf0, vuv, bap), dir_out,
                                      file_name)

            if return_dict:
                if len(output) > 0:
                    label_dict[file_name] = np.concatenate(output, axis=1)
                else:
                    label_dict[file_name] = None  # TODO: Or do not set at all.

        # Collect normalisation parameters.
        output_means = list()
        output_std_dev = list()

        load_flags = (self.load_sp, self.load_lf0, self.load_vuv,
                      self.load_bap)
        feature_sub_dirs = (self.dir_coded_sps, self.dir_lf0, self.dir_vuv,
                            self.dir_bap)
        feature_extensions = (self.sp_type, self.ext_lf0, self.ext_vuv,
                              self.ext_bap)
        feature_normalisers = (self.norm_params_ext_coded_sp,
                               self.norm_params_ext_lf0,
                               self.norm_params_ext_vuv,
                               self.norm_params_ext_bap)
        for load, feature_dir, normaliser, ext in\
                zip(load_flags, feature_sub_dirs, feature_normalisers,
                    feature_extensions):

            if not load:
                continue

            norm = normaliser.get_params()
            output_means.append(norm[0])
            output_std_dev.append(norm[1])

            if dir_out:
                norm_file_path = os.path.join(dir_out, feature_dir,
                                              file_id_list_name)

                if self.add_deltas and ext != self.ext_vuv:
                    has_name = file_id_list_name is not None \
                        and os.path.basename(file_id_list_name) != ""
                    if has_name:
                        norm_file_path += "-"
                    norm_file_path += "deltas"

                self.logger.info("Write norm_prams to {}".format(norm_file_path))
                normaliser.save(norm_file_path)

        if not self.add_deltas:
            if len(output_means) > 0:
                output_means = np.concatenate(output_means, axis=0)
                output_std_dev = np.concatenate(output_std_dev, axis=0)
            else:
                output_means = None
                output_std_dev = None

        if return_dict:
            return label_dict, output_means, output_std_dev
        else:
            return output_means, output_std_dev

    def _get_id_list(self, dir_in, file_id_list, id_list, file_ext):

        # Fill file_id_list by .wav files in dir_in if not given and set
        # an appropriate file_id_list_name.
        if id_list is None:
            id_list = list()
            filenames = glob.glob(os.path.join(dir_in, "*" + file_ext))
            for filename in filenames:
                id_list.append(os.path.splitext(os.path.basename(filename))[0])
            file_id_list_name = "all"
        else:
            file_id_list_name = os.path.splitext(os.path.basename(file_id_list))[0]

        return id_list, file_id_list_name

    def _create_directories(self, dir_out):
        if self.load_sp:
            makedirs_safe(os.path.join(dir_out, self.dir_coded_sps))
        if self.load_lf0:
            makedirs_safe(os.path.join(dir_out, self.dir_lf0))
        if self.load_vuv:
            makedirs_safe(os.path.join(dir_out, self.dir_vuv))
        if self.load_bap:
            makedirs_safe(os.path.join(dir_out, self.dir_bap))

    def _create_norm_params_extractors(self):
        if self.add_deltas:
            normaliser_class = MeanCovarianceExtractor
        else:
            normaliser_class = MeanStdDevExtractor

        self.norm_params_ext_coded_sp = normaliser_class()
        self.norm_params_ext_lf0 = normaliser_class()
        self.norm_params_ext_bap = normaliser_class()

        class NormaliserVUVDummy(object):
            """A dummy class to include VUV in the following loops."""
            def add_sample(self, *args):
                pass

            def save(self, *args):
                pass

            def get_params(self):
                return (0.0,), (1.0,)

        self.norm_params_ext_vuv = NormaliserVUVDummy()

    def save_output(self, features, dir_out, file_name):
        load_flags = (self.load_sp, self.load_lf0, self.load_vuv,
                      self.load_bap)
        feature_sub_dir = (self.dir_coded_sps, self.dir_lf0, self.dir_vuv,
                           self.dir_bap)
        feature_extensions = (self.sp_type, self.ext_lf0, self.ext_vuv,
                              self.ext_bap)
        feature_normalisers = (self.norm_params_ext_coded_sp,
                               self.norm_params_ext_lf0,
                               self.norm_params_ext_vuv,
                               self.norm_params_ext_bap)

        output = list()
        for load, feature, feature_dir, feature_ext, normaliser in zip(
                load_flags, features, feature_sub_dir, feature_extensions,
                feature_normalisers):

            if not load:
                continue

            file_name = os.path.basename(file_name)
            if self.add_deltas:

                if feature_dir != self.dir_vuv:
                    deltas = compute_deltas(feature)
                    double_deltas = compute_deltas(deltas)

                if dir_out is not None:
                    file_path = os.path.join(dir_out, feature_dir, file_name)
                    self._save_to_npz(file_path=file_path, features=feature,
                                      feature_name=feature_ext)
                    if feature_dir != self.dir_vuv:
                        self._save_to_npz(file_path=file_path, features=deltas,
                                          feature_name=feature_ext + "_deltas")
                        self._save_to_npz(
                            file_path=file_path,
                            features=double_deltas,
                            feature_name=feature_ext + "_double_deltas")

                if feature_dir != self.dir_vuv:
                    feature = np.concatenate((feature, deltas, double_deltas),
                                             axis=1)
            else:
                file_path = os.path.join(dir_out, feature_dir, file_name)
                self._save_to_npz(file_path=file_path, features=feature,
                                  feature_name=feature_ext)

            normaliser.add_sample(feature)

            output.append(feature)

        return output


def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "-a", "--dir_audio",
        help="Directory containing the audio (wav) files.",
        type=str, dest="dir_audio", required=True)
    parser.add_argument(
        "-s", "--sp_type",
        help="Type used to encode the spectral features into low dimensional "
             "representations.",
        type=str, dest="sp_type",
        choices=("mfcc", "mcep", "mgc", "mfbanks", "amp_sp", "log_amp_sp"),
        default="mcep")
    parser.add_argument(
        "-p", "--preemphasis",
        help="Preemphasis applied to the waveform.", type=float,
        dest="preemphasis", default=0.0)
    parser.add_argument(
        "--n_fft",
        help="Size of FFT window in librosa spectrum extraction.",
        type=int, dest="n_fft", default=None)
    parser.add_argument(
        "--window_length",
        help="Window length in ms used for STFT from time- to frequency-domain.",
        type=int, dest="window_length", default=None)
    parser.add_argument(
        "-n", "--num_coded_sps",
        help="Dimension of the frequency representation.",
        type=int, dest="num_coded_sps", default=60)
    parser.add_argument(
        "--hop_size",
        help="Hop size in ms used for STFT from time- to frequency-domain.",
        type=float, dest="hop_size_ms", default=5)
    parser.add_argument(
        "-i", "--file_id_list_path",
        help="Path to text file to read the ids of the files to process. "
             "Default uses all .wav files in the given audio_dir.",
        type=str, dest="file_id_list_path", default=None)
    parser.add_argument(
        "-o", "--dir_out",
        help="Output directory to store the labels. Within the output "
             "directory folders for each feature will be created.",
        type=str, dest="dir_out", required=True)
    parser.add_argument(
        "--add_deltas",
        help="Defines if features are augmented by their deltas and double "
             "deltas. Features will then be stored as a single file.",
        dest="add_deltas", action='store_const', const=True, default=False)
    parser.add_argument(
        "--no_sp", help="Omits extraction of spectral features.",
        dest="load_sp", action='store_false', default=True)
    parser.add_argument(
        "--no_lf0", help="Omits extraction of LF0 features.",
        dest="load_lf0", action='store_false', default=True)
    parser.add_argument(
        "--no_vuv", help="Omits extraction of V/UV features.",
        dest="load_vuv", action='store_false', default=True)
    parser.add_argument(
        "--no_bap", help="Omits extraction of BAP features.",
        dest="load_bap", action='store_false', default=True)

    args = parser.parse_args()

    dir_audio = os.path.abspath(args.dir_audio)
    dir_out = os.path.abspath(args.dir_out)

    # Read ids and select an appropriate file_id_list_name,
    # used to identify normalisation parameters of different subsets.
    if args.file_id_list_path is not None:
        file_id_list_path = os.path.abspath(args.file_id_list_path)
        with open(file_id_list_path) as f:
            id_list = f.readlines()
        id_list[:] = [s.strip(' \t\n\r') for s in id_list]
    else:
        id_list = None

    world_feat_gen = WorldFeatLabelGen(
        dir_out,
        add_deltas=args.add_deltas,
        preemphasis=args.preemphasis,
        num_coded_sps=args.num_coded_sps,
        n_fft=args.n_fft,
        win_length_ms=args.window_length,
        # num_bap=args.num_bap  # Not needed here because not used in gen_data.
        sp_type=args.sp_type,
        hop_size_ms=args.hop_size_ms,
        load_sp=args.load_sp,
        load_lf0=args.load_lf0,
        load_bap=args.load_bap,
        load_vuv=args.load_vuv)
    world_feat_gen.gen_data(
        dir_audio,
        dir_out=dir_out,
        file_id_list=args.file_id_list_path,
        id_list=id_list,
        return_dict=True)

    sys.exit(0)


if __name__ == "__main__":
    main()
