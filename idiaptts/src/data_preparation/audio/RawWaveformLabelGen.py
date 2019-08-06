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
import math
import os
import sys
from collections import OrderedDict
import numpy as np
import array
from timeit import default_timer as timer
from datetime import timedelta

# Third-party imports.
import pyworld
import scipy
from pydub import AudioSegment
from pydub.utils import get_array_type
import soundfile
import torch
from torch.utils.data import Dataset
import pysptk

# Local source tree imports.
from idiaptts.misc.normalisation.MeanStdDevExtractor import MeanStdDevExtractor
from idiaptts.misc.normalisation.MinMaxExtractor import MinMaxExtractor
from idiaptts.src.data_preparation.world.WorldFeatLabelGen import WorldFeatLabelGen
from idiaptts.misc.utils import makedirs_safe, interpolate_lin, compute_deltas


class RawWaveformLabelGen(Dataset):
    """Create world feat labels for .wav files."""

    logger = logging.getLogger(__name__)

    def __init__(self, norm_params=None, frame_rate_output_Hz=None, frame_size_ms=5, mu=None, silence_threshold_quantized=None):

        # Attributes.
        #self.id_list = id_list
        self.frame_rate_output_Hz = frame_rate_output_Hz
        self.frame_size_ms = frame_size_ms
        self.mu = mu
        self.silence_threshold_quantized = silence_threshold_quantized

        self.norm_params = norm_params

    # def __len__(self):
    #     return len(self.id_list)

    def __getitem__(self, file_path):
        """Return the preprocessed sample from the given id_name."""
        # file_path = self.id_list[item]

        sample = self.load_sample(file_path, self.frame_rate_output_Hz)
        sample = self.preprocess_sample(sample)

        return sample

    # def get_dims(self):
    #     labels_in, labels_out = self.__getitem__(0)
    #     return list(map(int, labels_in.shape[1:])), self.mu + 1
    #
    # def get_input(self, file_path):
    #     sample, raw = self.load_sample(file_path, self.add_deltas)
    #     sample, _ = self.preprocess_sample(raw, sample)
    #     return sample

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

    def preprocess_sample(self, sample):

        if self.mu is not None:
            # Mu law quantisation.
            sample = self.mu_law_companding(sample, self.mu)

            if self.silence_threshold_quantized is not None:
                start, end = RawWaveformLabelGen.start_and_end_indices(sample, self.silence_threshold_quantized)
                sample = sample[start:end]

            sample_dist = np.zeros((len(sample), self.mu + 1), dtype=np.float32)
            sample_dist[np.arange(len(sample)), sample] = 1.0

            return sample_dist
        else:
            sample = np.expand_dims(sample, -1).astype(np.float32)

        return sample

    @staticmethod
    def start_and_end_indices(quantized, silence_threshold=20):
        for start in range(quantized.size):
            if abs(quantized[start] - 127) > silence_threshold:
                break
        for end in range(quantized.size - 1, 1, -1):
            if abs(quantized[end] - 127) > silence_threshold:
                break

        assert abs(quantized[start] - 127) > silence_threshold
        assert abs(quantized[end] - 127) > silence_threshold

        return start, end

    def postprocess_sample(self, sample, norm_params=None):
        """
        Postprocess one sample. This function is used after inference of a network.

        :param sample:            The sample to post-process (T x C).
        :param norm_params:       Use this normalisation parameters instead of self.norm_params.
        :return:                  Post-processed sample.
        """

        # Convert one hot vector to index tensor.
        if self.mu is not None:
            sample = sample.argmax(axis=1)
            sample = self.mu_law_companding_reversed(sample, self.mu)

        if norm_params is not None:
            sample *= norm_params
        elif self.norm_params is not None:
            sample *= self.norm_params

        return sample

    @staticmethod
    def load_sample(file_path, frame_rate_output_Hz=None):
        """
        :param file_path:              Full path to the audio file.
        :param frame_rate_output_Hz:   Change the frame rate of the audio. Keep original if None.
        :return:                       Normalised raw waveform [-1, 1].
        """
        audio_seg = AudioSegment.from_file(file_path)
        bit_depth = audio_seg.sample_width * 8
        array_type = get_array_type(bit_depth)

        if frame_rate_output_Hz is not None and frame_rate_output_Hz != audio_seg.frame_rate:
            audio_seg = audio_seg.set_frame_rate(frame_rate_output_Hz)

        raw = np.array(array.array(array_type, audio_seg.raw_data), dtype=np.float64)
        raw /= math.pow(2, bit_depth) / 2  # Divide through maximum possible positive or negative number.

        return raw

    @staticmethod
    def mu_law_companding(raw, mu=255):
        raw = (((np.sign(raw) * np.log(1 + mu * np.abs(raw)) / np.log(1 + mu)) + 1.0) * (mu / 2.0)).astype(np.long)
        return raw

    @staticmethod
    def mu_law_companding_reversed(raw, mu=255):
        raw = raw / (mu / 2) - 1
        raw = np.sign(raw) / mu * (np.power(1.0 + mu, np.abs(raw)) - 1.0)
        return raw

    def set_normalisation_params(self, norm_params):
        self.norm_params = norm_params

        return self.norm_params

    def gen_data(self, dir_in, dir_out=None, file_id_list=None, id_list=None, add_deltas=False, return_dict=False):
        # TODO: Compute normalisation parameters here?
        raise NotImplementedError()


def main():
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-a", "--dir_audio", help="Directory containing the audio (wav) files.",
                        type=str, dest="dir_audio", required=True)
    parser.add_argument("-s", "--output_frequency", help="Desired frequency of all audio files [Hz].",
                        type=int, dest="output_frequency")
    # parser.add_argument("-f", "--frame_size_ms", help="Frame size of the labels.",
    #                     type=int, dest="frame_size_ms", default=5) # TODO: Use the frame size in pydubs from_file.
    parser.add_argument("-i", "--file_id_list_path",
                        help="Path to text file to read the ids of the files to process.\
                              Default uses all .wav files in the given audio_dir.",
                        type=str, dest="file_id_list_path", default=None)
    parser.add_argument("-o", "--dir_out", help="Output directory to store the wav files.",
                        type=str, dest="dir_out", required=True)
    parser.add_argument("-m", "--mu", help="If given mu-law quantisation is performed.",
                        type=int, dest="mu")

    # Parse arguments
    args = parser.parse_args()

    dir_audio = os.path.abspath(args.dir_audio)
    dir_out = os.path.abspath(args.dir_out)
    output_frequency = args.output_frequency
    mu = args.mu

    # Read ids and select an appropriate file_id_list_name,
    # used to identify normalisation parameters of different subsets.
    if args.file_id_list_path is not None:
        file_id_list_path = os.path.abspath(args.file_id_list_path)
        # Read which files to process.
        with open(file_id_list_path) as f:
            id_list = f.readlines()
        # Trim entries in-place.
        id_list[:] = [s.strip(' \t\n\r') for s in id_list]
        file_id_list_name = os.path.splitext(os.path.basename(file_id_list_path))[0]
    else:
        id_list = None
        file_id_list_name = "all"

    for index, id_name in enumerate(id_list):
        id_list[index] = os.path.join(dir_audio, id_name + ".wav")

    # Execute main functionality.
    raw_feat_gen = RawWaveformLabelGen(frame_rate_output_Hz=output_frequency, mu=mu)
    # label_dict, norm_first, norm_second = raw_feat_gen.gen_data(dir_audio, dir_out, args.file_id_list_path, id_list, add_deltas=False, return_dict=True)

    # DEBUG
    test_label = raw_feat_gen.load_sample(id_list[0], output_frequency)
    print(test_label[98:102])
    test_label = raw_feat_gen.preprocess_sample(test_label)
    print(test_label[98:102])
    test_label = raw_feat_gen.postprocess_sample(test_label)
    print(test_label[98:102])  # Slightly different because quantisation loses information.

    sys.exit(0)


if __name__ == "__main__":
    main()
