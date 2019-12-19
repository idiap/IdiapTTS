#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#

"""Module description:
   Create world feature labels for .wav files.
"""


# System imports.
import argparse
import glob
import logging
import numpy as np
import scipy
import os
import math
import sys
from collections import OrderedDict

# Third-party imports.
import pyworld
import soundfile
import pysptk
import librosa
import librosa.display
from nnmnkwii.postfilters import merlin_post_filter

# Local source tree imports.
from idiaptts.misc.normalisation.MeanStdDevExtractor import MeanStdDevExtractor
from idiaptts.misc.normalisation.MeanCovarianceExtractor import MeanCovarianceExtractor
from idiaptts.src.data_preparation.LabelGen import LabelGen
from idiaptts.misc.utils import makedirs_safe, interpolate_lin, compute_deltas
from idiaptts.misc.mlpg import MLPG


class WorldFeatLabelGen(LabelGen):
    """Create world feat labels for .wav files."""
    # TODO: Constants into hparams?
    f0_silence_threshold = 30
    lf0_zero = 0
    mgc_gamma = -1./3.
    preemphasis = 0.97

    dir_lf0 = "lf0"
    dir_vuv = "vuv"
    dir_bap = "bap"
    dir_deltas = "cmp"

    ext_lf0 = "lf0"
    ext_vuv = "vuv"
    ext_bap = "bap"
    ext_deltas = "cmp"

    logger = logging.getLogger(__name__)

    def __init__(self, dir_labels, add_deltas=False, sampling_fn=None, num_coded_sps=60, num_bap=1, sp_type="mcep", hop_size_ms=5,
                 load_sp=True, load_lf0=True, load_vuv=True, load_bap=True):
        """
        Constructor to use the class as a database.
        If add_delta is false labels have the dimension num_frames x (num_coded_sps + 3) [sp_type(num_coded_sps), lf0,
        vuv, bap(1)], otherwise the deltas and double deltas are added between the features resulting in
        num_frames x (3*num_coded_sps + 7) [sp_type(3*num_coded_sps), lf0(3*1), vuv, bap(3*1)].

        :param dir_labels:        While using it as a database dir_labels has to contain the prepared labels.
        :param add_deltas:        Determines if labels contain deltas and double deltas.
        :param sampling_fn:       Provide a function for up- or down-sampling the features during preprocessing.
        :param num_coded_sps:     Number of bins used to represent the coded spectral features.
        :param sp_type:           Type of the encoded spectral features e.g. mcep, mgc, mfbanks (mel-filter banks).
        :param hop_size_ms:       Hop size of FFT window in milliseconds.
        :param load_sp:           Whether to extract/load coded spectral features.
        :param load_lf0:          Whether to extract/load LF0.
        :param load_vuv:          Whether to extract/load V/UV flag.
        :param load_bap:          Whether to extract/load BAP.
        """

        # Save parameters.
        self.dir_labels = dir_labels  # Only used in __getitem__().
        self.add_deltas = add_deltas
        self.sampling_fn = sampling_fn
        self.num_coded_sps = num_coded_sps
        self.num_bap = num_bap
        self.sp_type = sp_type
        self.hop_size_ms = hop_size_ms
        self.load_sp = load_sp
        self.load_lf0 = load_lf0
        self.load_vuv = load_vuv
        self.load_bap = load_bap

        # Attributes.
        self.norm_params = None
        self.covs = [None] * 4  # Leave space for V/UV covariance even though it is never used.
        # self.cov_coded_sp = None
        # self.cov_lf0 = None
        # self.cov_bap = None
        self.dir_coded_sps = self.sp_type + str(self.num_coded_sps)
        self.dir_deltas += "_" + self.dir_coded_sps

    def __getitem__(self, id_name):
        """Return the preprocessed sample with the given id_name."""
        sample = self.load_sample(id_name,
                                  self.dir_labels,
                                  add_deltas=self.add_deltas,
                                  num_coded_sps=self.num_coded_sps,
                                  num_bap=self.num_bap,
                                  sp_type=self.sp_type,
                                  load_sp=self.load_sp,
                                  load_lf0=self.load_lf0,
                                  load_vuv=self.load_vuv,
                                  load_bap=self.load_bap)
        sample = self.preprocess_sample(sample)

        return sample

    @staticmethod
    def trim_end_sample(sample, length, reverse=False):
        """
        Trim the end of a sample by the given length. If reverse is True, the front of the sample is trimmed.
        This function is called after preprocess_sample.
        """
        if length == 0:
            return sample

        if reverse:
            return sample[length:, ...]
        else:
            return sample[:-length, ...]

    def _get_norm_params_subset(self, norm_params):
        mean, std_dev = norm_params

        num_processed_norm_params = 0
        num_sp_features = self.num_coded_sps * (3 if self.add_deltas else 1)
        if not self.load_sp:
            mean = mean[:num_sp_features]
            std_dev = std_dev[:num_sp_features]
        else:
            num_processed_norm_params += num_sp_features

        num_lf0_features = 3 if self.add_deltas else 1
        if not self.load_lf0:
            mean = mean[num_processed_norm_params:num_processed_norm_params + num_lf0_features]
            std_dev = std_dev[num_processed_norm_params:num_processed_norm_params + num_lf0_features]
        else:
            num_processed_norm_params += num_lf0_features

        if not self.load_vuv:
            mean = mean[num_processed_norm_params:num_processed_norm_params + 1]
            std_dev = std_dev[num_processed_norm_params:num_processed_norm_params + 1]
        else:
            num_processed_norm_params += 1

        num_bap_features = self.num_bap * (3 if self.add_deltas else 1)
        if not self.load_bap:
            mean = mean[num_processed_norm_params:num_processed_norm_params + num_bap_features]
            std_dev = std_dev[num_processed_norm_params:num_processed_norm_params + num_bap_features]
        else:
            num_processed_norm_params += num_bap_features

        return mean, std_dev

    def _get_norm_params(self, norm_params=None):
        if norm_params is not None:
            mean, std_dev = norm_params
        elif self.norm_params is not None:
            mean, std_dev = self.norm_params
        else:
            self.logger.error("Please give norm_params argument or call get_normaliations_params() before.")
            raise ValueError("Please give norm_params argument or call get_normalisation_params() before.")

        return mean, std_dev

    def preprocess_sample(self, sample, norm_params=None):
        """
        Normalise one sample (by default to 0 mean and variance 1). This function should be used within the
        batch loading of PyTorch.

        :param sample:            The sample to pre-process.
        :param norm_params:       Use this normalisation parameters instead of self.norm_params.
                                  Default distribution of norm_params for self.add_deltas=False is
                                  mgc, lf0, vuv, bap.
                                  Default distribution of norm_params for self.add_deltas=True is
                                  mgc, mgc deltas, mgc double deltas,
                                  lf0, lf0 deltas, lf0 double deltas,
                                  vuv,
                                  bap, bap deltas, bap double deltas.
        :return:                  Pre-processed sample.
        """
        mean, std_dev = self._get_norm_params(norm_params)
        sample = np.float32((sample - mean) / std_dev)

        if self.sampling_fn is not None:
            sample = self.sampling_fn(sample)

        return sample

    def postprocess_sample(self, sample, norm_params=None, apply_mlpg=True):
        """
        Postprocess one sample. This function is used after inference of a network.

        :param sample:            The sample to post-process.
        :param norm_params:       Use this normalisation parameters instead of self.norm_params.
        :param apply_mlpg:        Apply the MLPG algorithm on the post-processed sample.
        :return:                  Post-processed sample.
        """
        mean, std_dev = self._get_norm_params(norm_params)
        sample = np.copy((sample * std_dev) + mean)

        if self.add_deltas:
            output_list = list()
            mlpg = MLPG()

            num_processed_features = 0
            if self.load_sp:
                coded_sp_full = sample[:, :self.num_coded_sps * 3]
                num_processed_features += self.num_coded_sps * 3
                if apply_mlpg:
                    coded_sp = mlpg.generation(coded_sp_full, self.covs[0], self.covs[0].shape[0] // 3)
                else:
                    coded_sp = coded_sp_full[:, :self.num_coded_sps]
                output_list.append(coded_sp)

            if self.load_lf0:
                lf0_full = sample[:, num_processed_features:num_processed_features + 3]
                num_processed_features += 3
                if apply_mlpg:
                    lf0 = mlpg.generation(lf0_full, self.covs[1], self.covs[1].shape[0] // 3)
                    # lf0[vuv == 0] = -1e10
                    # lf0[lf0 <= math.log(20)] = -1e10  # Fix for WORLD vocoder which will cause memory error otherwise.
                else:
                    lf0 = lf0_full[:, 0:1]
                output_list.append(lf0)

            if self.load_vuv:
                vuv = sample[:, num_processed_features]
                num_processed_features += 1
                vuv[vuv < 0.5] = 0.0
                vuv[vuv >= 0.5] = 1.0
                vuv = vuv[:, None]
                output_list.append(vuv)

            if self.load_bap:
                bap_full = sample[:, -self.num_bap * 3:]

                if apply_mlpg:
                    bap = mlpg.generation(bap_full, self.covs[3], self.covs[3].shape[0] // 3)
                else:
                    bap = bap_full[:, 0:self.num_bap]
                output_list.append(bap)

            sample = np.concatenate(output_list, axis=1)

        return sample

    @staticmethod
    def load_sample(id_name, dir_out, add_deltas=False, num_coded_sps=60, num_bap=1, sp_type="mcep",
                    load_sp=True, load_lf0=True, load_vuv=True, load_bap=True):
        """
        Load world features from dir_out. It does not pre-process, use __getitem__ method instead.

        :param id_name:         Id of the sample.
        :param dir_out:         Directory containing the sample.
        :param add_deltas:      Determines if deltas and double deltas are expected.
        :param num_coded_sps:   Number of bins used to represent the coded spectral features.
        :param sp_type:         Type in which the coded spectral features are saved.
        :param load_sp:         Load spectral features defined by sp_type.
        :param load_lf0:        Load fundamental frequency.
        :param load_vuv:        Load voiced/unvoiced flag.
        :param load_bap:        Load band aperiodicity features.
        :return:                Numpy array with dimensions num_frames x len(coded_sp, lf0, vuv, bap).
        """
        id_name = os.path.splitext(os.path.basename(id_name))[0]
        logging.debug("Load WORLD " + ("deltas " if add_deltas else "") + "features for " + id_name)

        deltas_factor = 3 if add_deltas else 1
        dim_coded_sp = num_coded_sps * deltas_factor
        dim_lf0 = 1 * deltas_factor
        dim_vuv = 1
        dim_bap = num_bap * deltas_factor

        # If not all features are present also deltas features are saved separately.
        saved_as_cmp = add_deltas and load_sp and load_lf0 and load_vuv and load_bap

        output_list = list()
        if not saved_as_cmp:
            try:
                for load, feature_dir, feature_ext, feature_dim in\
                        zip((load_sp, load_lf0, load_vuv, load_bap),
                            (sp_type + str(num_coded_sps), WorldFeatLabelGen.dir_lf0,
                             WorldFeatLabelGen.dir_vuv, WorldFeatLabelGen.dir_bap),
                            (sp_type, WorldFeatLabelGen.ext_lf0,
                             WorldFeatLabelGen.ext_vuv, WorldFeatLabelGen.ext_bap),
                            (dim_coded_sp, dim_lf0, dim_vuv, dim_bap)):
                    if load:
                        path = os.path.join(dir_out, feature_dir, "{}.{}".format(id_name, feature_ext))
                        if add_deltas and feature_ext != WorldFeatLabelGen.ext_vuv:
                            path += "_deltas"
                        with open(path, 'rb') as f:
                            try:
                                feature = np.fromfile(f, dtype=np.float32)
                                labels = np.reshape(feature, [-1, feature_dim])
                            except ValueError as e:
                                logging.error("Cannot load labels from {}.".format(path))
                                raise e
                        output_list.append(labels)

                assert len(output_list) > 0, "At least one type of acoustic feature has to be loaded."
                labels = np.concatenate(output_list, axis=1)
                return labels
            except FileNotFoundError as e1:
                # Try to load from cmp folder. Reset output_list for it.
                output_list = list()

        path = os.path.join(dir_out,
                            "{}_{}{}".format(WorldFeatLabelGen.dir_deltas, sp_type, num_coded_sps),
                            "{}.{}".format(id_name, WorldFeatLabelGen.ext_deltas))
        with open(path, 'rb') as f:
            try:
                cmp = np.fromfile(f, dtype=np.float32)
                # cmp files always contain deltas.
                labels = np.reshape(cmp, [-1, 3 * (num_coded_sps + 1 + num_bap) + dim_vuv])
            except ValueError as e:
                logging.error("Cannot load labels from {}.".format(path))
                raise e

        if load_sp:
            output_list.append(labels[:, :dim_coded_sp])

        if load_lf0:
            output_list.append(labels[:, 3*num_coded_sps:3*num_coded_sps + dim_lf0])

        if load_vuv:
            output_list.append(labels[:, -3*num_bap-dim_vuv:-3*num_bap])

        if load_bap:
            if dim_bap == 3 * num_bap:
                output_list.append(labels[:, -3*num_bap:])
            else:
                output_list.append(labels[:, -3*num_bap:-3*num_bap + dim_bap])

        assert len(output_list) > 0, "At least one type of acoustic feature has to be loaded."
        labels = np.concatenate(output_list, axis=1)

        return labels

    def get_normalisation_params(self, dir_out, file_name=None):
        """
        Read the mean std_dev values from a file.
        Save them in self.norm_params.

        :param dir_out:       Directory containing the normalisation file.
        :param file_name:     Prefix of normalisation file (underscore "_" is expected as separator).
        :return:              Tuple of normalisation parameters (mean, std_dev).
        """

        # If not all features are present also deltas features are saved separately.
        saved_as_cmp = self.add_deltas and self.load_sp and self.load_lf0 and self.load_vuv and self.load_bap

        if not saved_as_cmp:# Collect all requested means and std_dev in a list.
            try:
                all_mean = list()
                all_std_dev = list()

                # full_path = os.path.join(dir_out, feature_dir,
                #                          "{}{}{}.bin".format(file_name + "_" if file_name is not None else "",
                #                                              feature_dir + "-" if self.add_deltas else "",
                #                                              MeanStdDevExtractor.file_name_appendix if self.add_deltas else MeanCovarianceExtractor.file_name_appendix))

                for load, feature_dir in zip((self.load_sp, self.load_lf0, self.load_vuv, self.load_bap),
                                             (self.dir_coded_sps, self.dir_lf0, self.dir_vuv, self.dir_bap)):
                    if load:
                        if feature_dir != self.dir_vuv:
                            if self.add_deltas:
                                full_path = os.path.join(dir_out, feature_dir,
                                                         "{}{}{}.bin".format(file_name + "-" if file_name is not None else "",
                                                                             feature_dir + "_deltas-",
                                                                             MeanCovarianceExtractor.file_name_appendix))
                            else:
                                full_path = os.path.join(dir_out, feature_dir,
                                                         "{}{}.bin".format(file_name + "-" if file_name is not None else "",
                                                                           MeanStdDevExtractor.file_name_appendix))

                            mean, std_dev = MeanStdDevExtractor.load(full_path)
                            all_mean.append(np.atleast_2d(mean))
                            all_std_dev.append(np.atleast_2d(std_dev))
                        else:
                            all_mean.append(np.atleast_2d(0.0))
                            all_std_dev.append(np.atleast_2d(1.0))

                # Save the concatenated normalisation parameters locally.
                self.norm_params = np.concatenate(all_mean, axis=1), np.concatenate(all_std_dev, axis=1)

                return self.norm_params
            except FileNotFoundError:
                # Try to load from the cmp folder.
                pass

        # Load the normalisation parameters.
        output_means = list()
        output_std_devs = list()
        for load, cov_idx, feature_dir in zip((self.load_sp, self.load_lf0, self.load_vuv, self.load_bap),
                                              range(4),
                                              (self.dir_coded_sps, self.dir_lf0, self.dir_vuv, self.dir_bap)):
            if load:
                if feature_dir != self.dir_vuv:
                    try:
                        path_new_style = os.path.join(dir_out, self.dir_deltas,
                                                      "{}{}-{}.bin".format(file_name + "-" if file_name is not None else "",
                                                                           feature_dir,
                                                                           MeanCovarianceExtractor.file_name_appendix))
                        mean, cov, std_dev = MeanCovarianceExtractor.load(path_new_style)
                    except FileNotFoundError as e1:
                        try:
                            # TODO: Remove legacy style path.
                            path = os.path.join(dir_out, self.dir_deltas,
                                                "{}{}_{}.bin".format(file_name + "-" if file_name is not None else "",
                                                                     MeanCovarianceExtractor.file_name_appendix,
                                                                     feature_dir))
                            mean, cov, std_dev = MeanCovarianceExtractor.load(path)
                            self.logger.warning("Found legacy style normalisation parameters at {}. "
                                                "Consider recreating features or renaming to {}"
                                                .format(path, path_new_style),
                                                DeprecationWarning)
                        except FileNotFoundError as e2:
                            raise FileNotFoundError([e1, e2])

                    # Assign to covariances.
                    self.covs[cov_idx] = cov
                    # Assign to output.
                    output_means.append(np.atleast_2d(mean))
                    output_std_devs.append(np.atleast_2d(std_dev))
                else:
                    output_means.append(np.atleast_2d(0.0))
                    output_std_devs.append(np.atleast_2d(1.0))

        self.norm_params = np.concatenate(output_means, axis=1)[0], np.concatenate(output_std_devs, axis=1)[0]

        return self.norm_params

    @staticmethod
    def fs_to_mgc_alpha(fs):
        """
        Convert sampling rate to appropriate MGC warping parameter.

        Code base on: Merlin's /misc/scripts/vocoder/world/extract_features_for_merlin.sh
        """
        return pysptk.util.mcepalpha(fs)
        # if fs == 16000:
        #     return 0.58
        # elif fs == 22050:
        #     return 0.65
        # elif fs == 44100:
        #     return 0.76
        # elif fs == 48000:
        #     return 0.77
        # else:
        #     raise NotImplementedError()

    @staticmethod
    def fs_to_frame_length(fs):
        """
        Convert sampling rate to frame length for STFT frame length.

        Code base on: Merlin's /misc/scripts/vocoder/world/extract_features_for_merlin.sh
        """
        return pyworld.get_cheaptrick_fft_size(fs)  # Better alternative.
        #
        # if fs == 16000 or fs == 22050 or fs == 24000:
        #     return 1024
        # elif fs == 44100 or fs == 48000:
        #     return 2048
        # else:
        #     raise NotImplementedError("The fs_to_frame_length method does not know how to handle {} Hz.".format(fs))

    @staticmethod
    def fs_to_num_bap(fs: int):
        return pyworld.get_num_aperiodicities(fs)

    @staticmethod
    def convert_to_world_features(sample, contains_deltas=False, num_coded_sps=60, num_bap=1):
        """Convert world acoustic features w/ or w/o deltas to WORLD understandable features."""

        deltas_factor = 3 if contains_deltas else 1
        assert sample.shape[1] == (num_coded_sps + 1 + num_bap) * deltas_factor + 1,\
            "WORLD requires all features to be present."

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
        """Convert from world features to a single feature vector with T x (|coded_sp|, |lf0|, |vuv|, |bap|) dim."""
        if lf0.ndim < 2:
            lf0 = lf0[:, None]
        if vuv.ndim < 2:
            vuv = vuv[:, None]
        if bap.ndim < 2:
            bap = bap[:, None]
        return np.concatenate((coded_sp, lf0, vuv, bap), axis=1)

    @staticmethod
    def framing(signal, sample_rate, frame_size_ms=None, frame_hop_ms=5):

        if frame_size_ms is None:
            frame_length = WorldFeatLabelGen.fs_to_frame_length(sample_rate)
        else:
            frame_size = frame_size_ms / 1000.
            frame_length = frame_size * sample_rate  # Convert from seconds to samples.

        frame_stride = frame_hop_ms / 1000.
        frame_step = frame_stride * sample_rate

        # Framing.
        signal_length = len(signal)
        frame_length = int(round(frame_length))
        frame_step = int(round(frame_step))
        num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame.

        pad_signal_length = num_frames * frame_step + frame_length
        z = np.zeros((pad_signal_length - signal_length))
        # Pad Signal to make sure that all frames have equal number of samples
        # without truncating any samples from the original signal.
        pad_signal = np.append(signal, z)

        indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(
            np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
        frames = pad_signal[indices.astype(np.int32, copy=False)]

        return frames

    @staticmethod
    def get_raw(audio_name: str, preemphasis: float = 0.97):
        """Extract the raw audio in [-1, 1] and apply pre-emphasis. 0.0 pre-emphasis means no pre-emphasis."""
        # librosa_raw = librosa.load(audio_name, sr=16000)  # raw in [-1, 1]
        # fs, raw = wavfile.read(audio_name)  # raw in [-32768, 32768]
        raw, fs = soundfile.read(audio_name)  # raw in [-1, 1]

        # Pre-emphasis
        raw = np.append(raw[0], raw[1:] - preemphasis * raw[:-1])

        return raw, fs

    # @staticmethod
    # def extract_mgc(raw: np.array, fs: int,
    #                 num_coded_sps: int = 60, frame_size_ms: int = None,
    #                 frame_hop_ms: int = 5, window_function: Callable = np.hanning,
    #                 mgc_alpha: float = None) -> np.array:
    #     """Extract MGC from raw [-1, 1] data with SPTK."""
    #
    #     if frame_size_ms is None:
    #         frame_length = WorldFeatLabelGen.fs_to_frame_length(fs)
    #     else:
    #         frame_length = int(frame_size_ms / 1000 * fs)
    #
    #     if mgc_alpha is None:
    #         mgc_alpha = WorldFeatLabelGen.fs_to_mgc_alpha(fs)
    #
    #     # Framing.
    #     padded_signal = np.pad(raw, pad_width=frame_length // 2, mode="reflect")
    #     frames = librosa.util.frame(padded_signal, frame_length, int(frame_hop_ms / 1000 * fs))
    #     # Windowing.
    #     frames *= window_function(frames.shape[1])
    #
    #     mgc = pysptk.mgcep(np.ascontiguousarray(frames.T),
    #                        order=num_coded_sps - 1,
    #                        alpha=mgc_alpha,
    #                        gamma=WorldFeatLabelGen.mgc_gamma,
    #                        eps=1.0e-8,
    #                        min_det=0.0,
    #                        etype=1,
    #                        itype=0)
    #
    #     return mgc.astype(np.float32, copy=False)

    @staticmethod
    def extract_mgc(amp_sp: np.array, fs: int = None, num_coded_sps: int = 60, mgc_alpha: float = None) -> np.array:
        """Extract MGC from the amplitude spectrum from SPTK."""

        if mgc_alpha is None:
            assert fs is not None, "Either sampling rate or mgc alpha has to be given."
            mgc_alpha = WorldFeatLabelGen.fs_to_mgc_alpha(fs)

        mgc = pysptk.mgcep(amp_sp,
                           order=num_coded_sps - 1,
                           alpha=mgc_alpha,
                           gamma=WorldFeatLabelGen.mgc_gamma,
                           eps=1.0e-8,
                           min_det=0.0,
                           etype=1,
                           itype=3)

        return mgc.astype(np.float32, copy=False)

    @staticmethod
    def extract_mcep(amp_sp: np.array, num_coded_sps: int, mgc_alpha: float) -> np.array:
        """Extract MCep from the amplitude spectrum with SPTK."""
        mcep = pysptk.mcep(amp_sp,
                           order=num_coded_sps - 1,
                           alpha=mgc_alpha,
                           eps=1.0e-8,
                           min_det=0.0,
                           etype=1,
                           itype=3)
        return mcep.astype(np.float32, copy=False)

    @staticmethod
    def librosa_extract_amp_sp(raw: np.array,
                               fs: int, n_fft: int = None,
                               hop_size_ms: int = 5,
                               win_length: int = None,
                               window: str = "hann",
                               center: bool = True,
                               pad_mode: str = 'reflect') -> np.array:
        """Extract amplitude spectrum from raw [-1, 1] signal. Parameters are explained in librosa.stft."""

        if n_fft is None:
            assert fs is not None, "Either fs or n_fft has to be given."
            n_fft = WorldFeatLabelGen.fs_to_frame_length(fs)

        amp_sp = np.abs(librosa.stft(raw, n_fft=n_fft, hop_length=int(hop_size_ms / 1000. * fs),
                                     win_length=win_length, center=center,
                                     window=window, pad_mode=pad_mode))

        return (amp_sp / np.sqrt(amp_sp.shape[0])).T  # T x n_fft

    @staticmethod
    def extract_mfbanks(raw: np.array = None, fs: int = 22050, amp_sp: np.array = None,
                        n_fft: int = None, hop_size_ms: int = 5, num_coded_sps: int = 80) -> np.array:
        """
        Extract Mel-filter banks using librosa.

        :param raw:            Raw audio signal in [-1, 1], ignored when amp_sp is given.
        :param fs:             Sampling rate
        :param amp_sp:         Amplitude spectrum, if not given it is extracted with librosa from the raw input.
        :param n_fft:          FFT length
        :param hop_size_ms:    Hop size in miliseconds.
        :param num_coded_sps:  Number of output Mel-filter banks
        :return:               Mel-filter banks as float32.
        """
        assert (n_fft is not None or amp_sp is not None), "Either FFT size has to be given or amplitude spectrogram."
        if amp_sp is None:
            assert raw is not None, "Either raw signal or amplitude spectrum must be given."
            amp_sp = WorldFeatLabelGen.librosa_extract_amp_sp(raw, fs, n_fft, hop_size_ms)
        mfbanks = librosa.feature.melspectrogram(sr=fs,
                                                 S=amp_sp.T,  # Use amplitude spectrum.
                                                 n_fft=n_fft,
                                                 hop_length=int(fs * hop_size_ms / 1000.0),
                                                 n_mels=num_coded_sps).T

        return mfbanks.astype(np.float32, copy=False)

    # @staticmethod
    # def extract_mfcc(raw: np.array, fs: int, amp_sp: np.array = None,
    #                  n_fft: int = None, hop_size_ms: int = 5, num_coded_sps = 12) -> np.array:
    #     # Using the default number (128) of mel bins.
    #     mel_sp = librosa.feature.melspectrogram(y=raw,  # Ignored when amp_sp not None.
    #                                             sr=fs,
    #                                             S=amp_sp.T if amp_sp is not None else None,  # Use amplitude spectrum.
    #                                             n_fft=n_fft,
    #                                             hop_length=int(fs * hop_size_ms / 1000.0))
    #     log_pow_mel_sp = librosa.power_to_db(np.square(mel_sp), top_db=None)
    #     mfcc = librosa.feature.mfcc(sr=fs, S=log_pow_mel_sp, n_mfcc=num_coded_sps).T
    #
    #     return mfcc

    @staticmethod
    def mcep_to_amp_sp(mcep: np.array, fs: int, alpha: float = None):
        """Convert MCep back to amplitude spectrum using SPTK."""
        if alpha is None:
            alpha = WorldFeatLabelGen.fs_to_mgc_alpha(fs)
        amp_sp = pysptk.mgc2sp(np.ascontiguousarray(mcep, dtype=np.float64),
                               alpha=alpha,
                               gamma=0.0,
                               fftlen=WorldFeatLabelGen.fs_to_frame_length(fs))
        return np.exp(amp_sp.real.astype(np.float32, copy=False))

    @staticmethod
    def mgc_to_amp_sp(mgc: np.array, fs: int, alpha: float = None, gamma: float = None, n_fft: int = None):
        """Convert MGCs back to amplitude spectrum using SPTK."""
        if alpha is None:
            alpha = WorldFeatLabelGen.fs_to_mgc_alpha(fs)
        if gamma is None:
            gamma = WorldFeatLabelGen.mgc_gamma
        if n_fft is None:
            n_fft = WorldFeatLabelGen.fs_to_frame_length(fs)
        amp_sp = pysptk.mgc2sp(np.ascontiguousarray(mgc, dtype=np.float64),
                               alpha=alpha,
                               gamma=gamma,
                               fftlen=n_fft)

        # WORLD expects spectrum divided by number of final bins, but SPTK does not divide it.
        return np.exp(amp_sp.real.astype(np.float32, copy=False))

    @staticmethod
    def amp_sp_to_raw(amp_sp: np.array, fs: int, hop_size_ms: int = 5, preemphasis: float = 0.97):
        """
        Transform the amplitude spectrum into the waveform with Griffin-Lim.
        The amplitude spectrum has to have the pitch information. Using amplitude spectrum which was extracted with
        pitch aligned windows (as WORLD does it) will not work.
        """
        raw = librosa.griffinlim(amp_sp.T * np.sqrt(amp_sp.shape[1]), hop_length=int(fs * hop_size_ms / 1000.))
        return scipy.signal.lfilter([1], [1, -preemphasis], raw)  # de-preemphasis

    @staticmethod
    def world_features_to_raw(amp_sp: np.array, lf0: np.array, vuv: np.array, bap: np.array, fs: int, n_fft: int = None,
                              f0_silence_threshold: int = None, lf0_zero: float = None, preemphasis: float = 0.97):
        """Using the world vocoder to generate the waveform."""
        if f0_silence_threshold is None:
            f0_silence_threshold = WorldFeatLabelGen.f0_silence_threshold
        if lf0_zero is None:
            lf0_zero = WorldFeatLabelGen.lf0_zero
        if n_fft is None:
            n_fft = WorldFeatLabelGen.fs_to_frame_length(fs)

        pow_sp = np.square(amp_sp, dtype=np.float64)

        f0 = np.exp(lf0, dtype=np.float64)
        vuv[f0 < f0_silence_threshold] = 0  # WORLD throws an error for too small f0 values.
        f0[vuv == 0] = lf0_zero
        if f0.ndim > 1:
            assert f0.shape[1:] == (1,) * (f0.ndim - 1), "F0 should have only one dimension at this stage."
            f0 = f0.squeeze()

        if bap.ndim < 2:
            bap = bap.reshape(-1, 1)

        ap = pyworld.decode_aperiodicity(np.ascontiguousarray(bap, np.float64), fs, n_fft)

        raw = pyworld.synthesize(f0, pow_sp, ap, fs).astype(np.float32, copy=False)  # Inplace conversion, if possible.
        return scipy.signal.lfilter([1], [1, -preemphasis], raw)  # de-preemphasis

    @staticmethod
    def mfbanks_to_amp_sp(coded_sp: np.array, fs: int, n_fft: int = None):
        """Convert Mel-filter banks back to amplitude spectrum. This does not work well. Use an SSRN instead."""
        if n_fft is None:
            n_fft = WorldFeatLabelGen.fs_to_frame_length(fs)
        amp_sp = librosa.feature.inverse.mel_to_stft(coded_sp.T, sr=fs, n_fft=n_fft, power=1.0, norm=None).T
        return amp_sp * amp_sp.shape[1]

    @staticmethod
    def world_extract_features(raw: np.array,
                               fs: int,
                               hop_size_ms: int,
                               f0_silence_threshold: int = None,
                               lf0_zero: float = None):
        """Extract WORLD features """
        if f0_silence_threshold is None:
            f0_silence_threshold = WorldFeatLabelGen.f0_silence_threshold
        if lf0_zero is None:
            lf0_zero = WorldFeatLabelGen.lf0_zero

        f0, pow_sp, ap = pyworld.wav2world(raw, fs, frame_period=hop_size_ms)  # Gives power spectrum in [0, 1]

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
                         preemphasis: float = 0.97, sp_type: str = "mcep", num_coded_sps: int = 40,
                         load_sp: bool = True, load_lf0: bool = True, load_vuv: bool = True, load_bap: bool = True,
                         hop_size_ms: int = 5, f0_silence_threshold: int = None, lf0_zero: float = None):
        """
        Extract acoustic features from a single audio file.
        This function is called from the gen_data function.
        """

        # Load raw audio file.
        audio_name = os.path.join(dir_in, file_name + "." + file_ext)
        raw, fs = WorldFeatLabelGen.get_raw(audio_name, preemphasis)

        extr_features_msg = ""
        lf0, vuv, bap = None, None, None
        if sp_type in ["mcep", "mgc", "amp_sp"] or load_lf0 or load_vuv or load_bap:
            amp_sp, lf0, vuv, bap = WorldFeatLabelGen.world_extract_features(raw,
                                                                             fs,
                                                                             hop_size_ms,
                                                                             f0_silence_threshold,
                                                                             lf0_zero)

            # Compute the deltas and double deltas.
            if load_lf0:
                extr_features_msg += " WORLD lf0,"

            if load_vuv:
                # Throw a warning when less then 5% of all frames are unvoiced.
                if vuv.sum() / len(vuv) < 0.05:
                    logging.warning("Detected only {:.0f}% [{}/{}] unvoiced frames in {}."
                                    .format(vuv.sum() / len(vuv) * 100.0, int(vuv.sum()), len(vuv), file_name))
                extr_features_msg += " WORLD vuv,"
                # V/UV never has deltas features.

            if load_bap:
                extr_features_msg += " WORLD {}bap,".format(bap.shape[1])

        coded_sp = None
        if load_sp:
            if sp_type == "mcep":
                coded_sp = WorldFeatLabelGen.extract_mcep(amp_sp,
                                                          num_coded_sps=num_coded_sps,
                                                          mgc_alpha=WorldFeatLabelGen.fs_to_mgc_alpha(fs))
                assert len(coded_sp) == len(lf0), "Requires testing. Possibly trimming is a solution."
                extr_features_msg = "WORLD {}{},{}".format(num_coded_sps, sp_type, extr_features_msg)
            elif sp_type == "mgc":
                coded_sp = WorldFeatLabelGen.extract_mgc(amp_sp,
                                                         fs=fs,
                                                         num_coded_sps=num_coded_sps)
                extr_features_msg = "WORLD {}{},{}".format(num_coded_sps, sp_type, extr_features_msg)
            elif sp_type == "mfbanks":
                coded_sp = WorldFeatLabelGen.extract_mfbanks(raw,
                                                             fs=fs,
                                                             n_fft=WorldFeatLabelGen.fs_to_frame_length(fs),
                                                             hop_size_ms=hop_size_ms,
                                                             num_coded_sps=num_coded_sps)
                extr_features_msg = "Librosa {}{},{}".format(num_coded_sps, sp_type, extr_features_msg)
            elif sp_type == "amp_sp":
                coded_sp = amp_sp
                extr_features_msg = "WORLD {}{},{}".format(amp_sp.shape[1], sp_type, extr_features_msg)

        # Log some debug information.
        file_name = os.path.basename(file_name)  # Remove speaker.
        logging.info("Extracted ({}) features from {} at {} Hz with {} ms frame hop."
                     .format(extr_features_msg.strip().strip(','), file_name, fs, hop_size_ms))

        # # DEBUG
        # import matplotlib.pyplot as plt
        # amp_sp = np.sqrt(pow_sp)  # * (2 ** 16) / 2.
        # hop_size_frames = int(fs * hop_size_ms / 1000.0)
        # librosa_amp_sp = np.abs(librosa.core.stft(y=raw,
        #                                           n_fft=WorldFeatLabelGen.fs_to_frame_length(fs),
        #                                           hop_length=hop_size_frames)).T
        # plt.figure(figsize=(10, 4))
        # librosa.display.specshow(librosa.power_to_db(librosa_amp_sp.T, ref=np.max), sr=fs, hop_length=hop_size_ms)
        # plt.colorbar()
        # plt.title("librosa amp sp")
        # plt.show()
        #
        # plt.figure()
        # librosa.display.specshow(librosa.power_to_db(amp_sp.T, ref=np.max), sr=fs, hop_length=hop_size_ms)
        # plt.colorbar()
        # plt.title("WORLD amp sp")
        # plt.show()

        # frame_idx = 200
        # plt.plot(librosa.power_to_db(pow_sp.T * pow_sp.shape[1], top_db=200)[:, frame_idx], "b-", linewidth=2.0,
        #          label="WORLD amplitude spectrum 20log|X(w)|")
        # plt.plot(librosa.amplitude_to_db(librosa_amp_sp.T, top_db=200)[:, frame_idx], "r-", linewidth=2.0,
        #          label="Librosa amplitude spectrum 20log|X(w)|")
        # # plt.plot(20/np.log(20)*envelope, "r-", lindewidth=3.0, label="Reconstruction")
        # plt.xlabel("frequency bin")
        # plt.ylabel("log amplitude")
        # plt.legend()
        # plt.show()

        return coded_sp, lf0, vuv, bap

    @staticmethod
    def decode_sp(coded_sp: np.array, sp_type: str = "mcep", fs: int = None, alpha: float = None, mgc_gamma:
                  float = None, n_fft: int = None, post_filtering: bool = False):

        if post_filtering:
            if sp_type in ["mcep", "mgc"]:
                coded_sp = merlin_post_filter(coded_sp, WorldFeatLabelGen.fs_to_mgc_alpha(fs))
            else:
                logging.warning("Post-filtering only implemented for cepstrum features.")

        if sp_type == "mcep":
            return WorldFeatLabelGen.mcep_to_amp_sp(coded_sp, fs, alpha)
        elif sp_type == "mgc":
            return WorldFeatLabelGen.mgc_to_amp_sp(coded_sp, fs, alpha, mgc_gamma, n_fft)
        elif sp_type == "mfbanks":
            return WorldFeatLabelGen.mfbanks_to_amp_sp(coded_sp, fs, n_fft)
        else:
            raise NotImplementedError("Unknown feature type {}. No decoding method available.".format(sp_type))

    def gen_data(self, dir_in, dir_out=None, file_id_list="", file_ext="wav", id_list=None, return_dict=False):
        """
        Prepare acoustic features from audio files. Which features are extracted are determined by the parameters
        given in the constructor. The self.load_* flags determine if that features is extracted and the self.sp_type
        determines the type of coded spectrum representation.

        :param dir_in:         Directory where the audio files are stored for each utterance to process.
        :param dir_out:        Main directory where the labels and normalisation parameters are saved to subdirectories.
                               If None, labels are not saved.
        :param file_id_list:   Name of the file containing the ids. Normalisation parameters are saved using
                               this name to differentiate parameters between subsets.
        :param file_ext:       Extension of all audio files.
        :param id_list:        The list of utterances to process.
                               Should have the form uttId1 \\n uttId2 \\n ...\\n uttIdN.
                               If None, all file in audio_dir are used.
        :param return_dict:    If true, returns an OrderedDict of all samples as first output_return_dict.
        :return:               Returns two normalisation parameters as tuple. If return_dict is True it returns
                               all processed labels in an OrderedDict followed by the two normalisation parameters.
        """

        # Fill file_id_list by .wav files in dir_in if not given and set an appropriate file_id_list_name.
        if id_list is None:
            id_list = list()
            filenames = glob.glob(os.path.join(dir_in, "*" + file_ext))
            for filename in filenames:
                id_list.append(os.path.splitext(os.path.basename(filename))[0])
            file_id_list_name = "all"
        else:
            file_id_list_name = os.path.splitext(os.path.basename(file_id_list))[0]

        # .cmp files contain all features with their deltas and double deltas.
        # If not all features are present save them separately instead.
        save_as_cmp = self.add_deltas and self.load_sp and self.load_lf0 and self.load_vuv and self.load_bap

        # Create directories in dir_out if it is given.
        if dir_out is not None:
            if save_as_cmp:
                makedirs_safe(os.path.join(dir_out, self.dir_deltas))
            else:
                if self.load_sp:
                    makedirs_safe(os.path.join(dir_out, self.dir_coded_sps))
                if self.load_lf0:
                    makedirs_safe(os.path.join(dir_out, self.dir_lf0))
                if self.load_vuv:
                    makedirs_safe(os.path.join(dir_out, self.dir_vuv))
                if self.load_bap:
                    makedirs_safe(os.path.join(dir_out, self.dir_bap))

        # Create the return dictionary if required.
        if return_dict:
            label_dict = OrderedDict()

        if self.add_deltas:
            # Create normalisation computation units.
            norm_params_ext_coded_sp = MeanCovarianceExtractor()
            norm_params_ext_lf0 = MeanCovarianceExtractor()
            norm_params_ext_bap = MeanCovarianceExtractor()
        else:
            # Create normalisation computation units.
            norm_params_ext_coded_sp = MeanStdDevExtractor()
            norm_params_ext_lf0 = MeanStdDevExtractor()
            norm_params_ext_bap = MeanStdDevExtractor()

        class NormaliserVUVDummy(object):
            """A dummy class to include VUV in the following loops."""
            def add_sample(self, *args):
                pass

            def save(self, *args):
                pass

            def get_params(self):
                return (0.0,), (1.0,)

        norm_params_ext_vuv = NormaliserVUVDummy()

        logging.info("Extract acoustic features{} for ".format("" if not self.add_deltas else " with deltas")
                     + "[{0}]".format(", ".join(str(i) for i in id_list)))

        # Extract feature for each
        for file_name in id_list:
            # Extract acoustic features from an audio file.
            coded_sp, lf0, vuv, bap = self.extract_features(dir_in, file_name, file_ext,
                                                            preemphasis=self.preemphasis,
                                                            sp_type=self.sp_type,
                                                            num_coded_sps=self.num_coded_sps,
                                                            load_sp=self.load_sp,
                                                            load_lf0=self.load_lf0,
                                                            load_vuv=self.load_vuv,
                                                            load_bap=self.load_bap,
                                                            hop_size_ms=self.hop_size_ms,
                                                            f0_silence_threshold=WorldFeatLabelGen.f0_silence_threshold,
                                                            lf0_zero=WorldFeatLabelGen.lf0_zero)

            output = list()
            for load, feature, feature_dir, feature_ext, normaliser in\
                    zip((self.load_sp, self.load_lf0, self.load_vuv, self.load_bap),
                        (coded_sp, lf0, vuv, bap),
                        (self.dir_coded_sps, self.dir_lf0, self.dir_vuv, self.dir_bap),
                        (self.sp_type, self.ext_lf0, self.ext_vuv, self.ext_bap),
                        (norm_params_ext_coded_sp, norm_params_ext_lf0, norm_params_ext_vuv, norm_params_ext_bap)):
                if load:  # Check if feature should be loaded.
                    file_name = os.path.basename(file_name)
                    if self.add_deltas:  # Add deltas if requested.
                        if feature_ext != "vuv":
                            deltas, double_deltas = compute_deltas(feature)
                            feature = np.concatenate((feature, deltas, double_deltas), axis=1)
                        if dir_out is not None and not save_as_cmp:
                            # Not all features for cmp are present so save the labels separately in deltas directory.
                            feature.tofile(os.path.join(dir_out,
                                                        feature_dir,
                                                        "{}.{}{}".format(
                                                            file_name,
                                                            feature_ext,
                                                            "_deltas" if feature_ext != self.ext_vuv else "")))
                    else:
                        # Save features without deltas in their respective subdirectory.
                        feature.tofile(os.path.join(dir_out, feature_dir, "{}.{}".format(file_name, feature_ext)))

                    # Add sample to normalisation computation unit.
                    normaliser.add_sample(feature)

                    # Add to list of output features.
                    output.append(feature)

            # Save into a single file if all features are present (only when deltas are added).
            if dir_out is not None and save_as_cmp:
                # Combine them to a single feature sample.
                labels = np.concatenate(output, axis=1)
                labels.tofile(os.path.join(dir_out, self.dir_deltas, "{}.{}".format(os.path.basename(file_name),
                                                                                    self.ext_deltas)))

            if return_dict:
                # Save into return dictionary.
                label_dict[file_name] = np.concatenate(output, axis=1) if len(output) > 0 else None
        # END feature extraction loop.

        # Collect normalisation parameters.
        output_means = list()
        output_std_dev = list()
        for load, feature_dir, normaliser, ext in\
                zip((self.load_sp, self.load_lf0, self.load_vuv, self.load_bap),
                    (self.dir_coded_sps, self.dir_lf0, self.dir_vuv, self.dir_bap),
                    (norm_params_ext_coded_sp, norm_params_ext_lf0, norm_params_ext_vuv, norm_params_ext_bap),
                    (self.dir_coded_sps, self.ext_lf0, self.ext_vuv, self.ext_bap)):
            if load:  # Check if feature was extracted.
                # Collect the normalisation parameters to return them.
                norm = normaliser.get_params()
                output_means.append(norm[0])
                output_std_dev.append(norm[1])

                if dir_out:
                    # Select the correct output directory to save the normalisation parameters.
                    if self.add_deltas:
                        modified_file_id_list_name = file_id_list_name if (file_id_list_name is None
                                                                           or file_id_list_name == "")\
                                                                       else file_id_list_name + "-"
                        if save_as_cmp:
                            norm_file_path = os.path.join(
                                dir_out,
                                self.dir_deltas,
                                "{}{}".format(modified_file_id_list_name, ext))
                        else:
                            if ext == self.ext_vuv:  # Special case; VUV should never be saved with deltas ending.
                                norm_file_path = os.path.join(dir_out,
                                                              feature_dir,
                                                              "{}{}".format(modified_file_id_list_name, ext))
                            else:
                                norm_file_path = os.path.join(dir_out,
                                                              feature_dir,
                                                              "{}{}_deltas".format(modified_file_id_list_name, ext))
                    else:
                        norm_file_path = os.path.join(dir_out, feature_dir, file_id_list_name)
                    self.logger.info("Write norm_prams to {}".format(norm_file_path))
                    normaliser.save(norm_file_path)

        if not self.add_deltas:
            output_means = np.concatenate(output_means, axis=0) if len(output_means) > 0 else None
            output_std_dev = np.concatenate(output_std_dev, axis=0) if len(output_std_dev) > 0 else None

        if return_dict:
            # Return dict of labels for all utterances.
            return label_dict, output_means, output_std_dev
        else:
            return output_means, output_std_dev


def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-a", "--dir_audio", help="Directory containing the audio (wav) files.",
                        type=str, dest="dir_audio", required=True)
    parser.add_argument("-s", "--sp_type",
                        help="Type used to encode the spectral features into low dimensional representations.",
                        type=str, dest="sp_type", choices=("mfcc", "mcep", "mgc", "mfbanks"), default="mcep")
    parser.add_argument("-n", "--num_coded_sps", help="Dimension of the frequency representation.",
                        type=int, dest="num_coded_sps", default=60)
    parser.add_argument("--hop_size", help="Hop size in ms used for STFT from time- to frequency-domain.",
                        type=int, dest="hop_size_ms", default=5)
    parser.add_argument("-i", "--file_id_list_path",
                        help="Path to text file to read the ids of the files to process.\
                              Default uses all .wav files in the given audio_dir.",
                        type=str, dest="file_id_list_path", default=None)
    parser.add_argument("-o", "--dir_out", help="Output directory to store the labels."
                                                "Within the output directory folders for each feature will be created.",
                        type=str, dest="dir_out", required=True)
    parser.add_argument("--add_deltas", help="Defines if features are augmented by their deltas and double deltas."
                                             "Features will then be stored as a single file.",
                        dest="add_deltas", action='store_const', const=True, default=False)

    # Parse arguments
    args = parser.parse_args()

    dir_audio = os.path.abspath(args.dir_audio)
    dir_out = os.path.abspath(args.dir_out)

    # Read ids and select an appropriate file_id_list_name,
    # used to identify normalisation parameters of different subsets.
    if args.file_id_list_path is not None:
        file_id_list_path = os.path.abspath(args.file_id_list_path)
        # Read which files to process.
        with open(file_id_list_path) as f:
            id_list = f.readlines()
        # Trim entries in-place.
        id_list[:] = [s.strip(' \t\n\r') for s in id_list]
    else:
        id_list = None

    # Execute main functionality.
    world_feat_gen = WorldFeatLabelGen(dir_out,
                                       add_deltas=args.add_deltas,
                                       num_coded_sps=args.num_coded_sps,
                                       # num_bap=args.num_bap  # Is not needed here because it is not used in gen_data.
                                       sp_type=args.sp_type,
                                       hop_size_ms=args.hop_size_ms)
    world_feat_gen.gen_data(dir_audio,
                            dir_out=dir_out,
                            file_id_list=args.file_id_list_path,
                            id_list=id_list,
                            return_dict=True)

    sys.exit(0)


if __name__ == "__main__":
    main()
