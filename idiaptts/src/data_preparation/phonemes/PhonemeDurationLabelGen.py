#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#

"""Module description:
   Generate phoneme durations from different aligned labels.
"""

# System imports.
import argparse
from collections import OrderedDict
import glob
import logging
import numpy as np
import os
import sys
from typing import Union, Any, List, Optional, cast

# Third-party imports.

# Local source tree imports.
from idiaptts.misc.normalisation.MeanStdDevExtractor import MeanStdDevExtractor
from idiaptts.misc.utils import makedirs_safe
from idiaptts.src.data_preparation.LabelGen import LabelGen
from idiaptts.src.data_preparation.NpzDataReader import NpzDataReader


class PhonemeDurationLabelGen(NpzDataReader, LabelGen):
    """Load phoneme durations from state-aligned HTK labels."""

    # Extension of created duration labels.
    ext_durations = ".dur"
    # Number of states per phoneme in HTK labels.
    num_states = 5
    # Divisor for phoneme timings in HTK labels. Only used in gen_data.
    min_phoneme_length = 50000
    # Divisor for phoneme timings in MFA labels. Only used in gen_data.
    frame_length_sec = 0.005

    logger = logging.getLogger(__name__)

    class Config(NpzDataReader.Config):

        def __init__(self, *args,
                     norm_type: str = NpzDataReader.Config.NormType.MEAN_STDDEV,
                     load_as_matrix: bool = False,
                     pad_mode: str = 'constant',
                     label_type: str = 'full_state_align',
                     **kwargs) -> None:
            super().__init__(*args, norm_type=norm_type, **kwargs)

            self.load_as_matrix = load_as_matrix
            if load_as_matrix:
                assert pad_mode == 'edge', \
                     "Use \'edge\' pad_mode for duration matrix."
                assert norm_type == NpzDataReader.Config.NormType.NONE, \
                    "Duration matrix should not be normalised."

            self.label_type = label_type

        def create_reader(self):
            reader = PhonemeDurationLabelGen(self)

            reader.get_normalisation_params()

            return reader

    def __init__(self, *args, **kwargs):

        if len(args) == 1 \
                and isinstance(args[0], PhonemeDurationLabelGen.Config):
            config = args[0]

            super().__init__(config)

            self.load_as_matrix = config.load_as_matrix
            self.label_type = config.label_type

            self.legacy_getitem = False
        else:
            # LEGACY support
            if len(args) > 0:
                dir_labels = args[0]
                if len(args) > 1:
                    self.load_as_matrix = args[1]
                else:
                    self.load_as_matrix = kwargs.get("load_as_matrix", False)
            else:
                dir_labels = kwargs["dir_labels"]
                self.load_as_matrix = kwargs.get("load_as_matrix", False)

            super().__init__(
                config=PhonemeDurationLabelGen._get_reader_config(
                    dir_labels, self.load_as_matrix,
                    label_type="full_state_align"))
            self.label_type = "full_state_align"

            self.legacy_getitem = True

        if type(self.directory) in [tuple, list]:
            self.dir_labels = self.directory[0]
        else:
            self.dir_labels = self.directory

    @staticmethod
    def _get_reader_config(dir_labels, load_as_matrix, label_type):
        if load_as_matrix:
            pad_mode = 'edge'
            norm_type = NpzDataReader.Config.NormType.NONE
        else:
            pad_mode="constant"
            norm_type = NpzDataReader.Config.NormType.MEAN_STDDEV

        return PhonemeDurationLabelGen.Config(name="durations",
                                              directory=dir_labels,
                                              load_as_matrix=load_as_matrix,
                                              pad_mode=pad_mode,
                                              norm_type=norm_type,
                                              label_type=label_type)

    def __getitem__(self, id_name):
        """Return the preprocessed sample with the given id_name."""
        sample_dict = super().__getitem__(id_name)

        if self.legacy_getitem:
            # LEGACY support
            return sample_dict[self.output_names[0]]
        else:
            return sample_dict

    def load(self, id_name: str):
        logging.debug("Load duration for " + id_name)
        try:
            sample = super().load(id_name)

        except FileNotFoundError:
            # LEGACY support
            id_name = os.path.splitext(os.path.basename(id_name))[0]
            file_path = os.path.join(self.directory[0], id_name)

            try:
                archive = np.load(file_path + ".npz")
                sample = archive["dur"]
            except FileNotFoundError:
                file_path += self.ext_durations
                with open(file_path, 'r') as f:
                    sample = np.fromfile(f, dtype=np.float32)

                # TODO: Change to label_state_align as in PhonemeLabelGen.
                if self.label_type == "full_state_align":
                    sample = np.reshape(sample, [-1, self.num_states])
                elif self.label_type == "mfa":
                    sample = sample[:, None]
                else:
                    raise NotImplementedError("Unknown label type {}.".format(
                        self.label_type))

            if sample.ndim < 2:
                sample = sample[:, None]

        if self.load_as_matrix:
            sample = self.convert_to_matrix(sample)

        return sample

    def convert_to_matrix(self, sample):
        # Sample is T x 5 in frames of 50000 * 100 ns = 5 ms.
        # TODO: Has to be adapted for different frame shift?
        return self.durations_to_hard_attention_matrix(
            sample.sum(axis=1).astype(np.int))

    @staticmethod
    def durations_to_hard_attention_matrix(durations):
        '''
        Code from https://github.com/CSTR-Edinburgh/ophelia/blob/a754abdf54986c31c43c363db1a5f850df06fdc6/utils.py#L188

        Take array of durations, return selection matrix to replace A in
        attention mechanism.

        E.g.:
        durations_to_hard_attention_matrix(np.array([3,0,1,2]))
        [[1. 1. 1. 0. 0. 0.]
         [0. 0. 0. 0. 0. 0.]
         [0. 0. 0. 1. 0. 0.]
         [0. 0. 0. 0. 1. 1.]]
        '''
        num_phones = len(durations)
        num_frames = durations.sum()
        A = np.zeros((num_frames, num_phones), dtype=np.float32)
        start = 0
        for (i, dur) in enumerate(durations):
            end = start + dur
            A[start:end, i] = 1.0
            start = end
        assert (A.sum(axis=1) == 1.0).all()
        assert (A.sum(axis=0) == durations).all()
        return A

    @staticmethod
    def load_sample(id_name, dir_out, label_type="full_state_align"):
        """
        Load durations from dir_out.

        :param id_name:  Id of the sample.
        :param dir_out:  Directory containing the sample.
        :return:         Numpy array with dimensions
                         num_frames x self.num_states.
        """

        datareader = PhonemeDurationLabelGen._get_reader_config(
            dir_out, load_as_matrix=False, label_type=label_type).create_reader()
        return datareader.load(id_name)

    @staticmethod
    def gen_data(dir_in, dir_out=None, file_id_list="", id_list=None,
                 label_type="full_state_align", return_dict=False):
        """
        Prepare durations from HTK labels (forced-aligned).
        Each numpy array has the dimension
        num_phonemes x PhonemeDurationLabelGen.num_states (default
        num_state=5).

        :param dir_in:         Directory where the HTK label files are
                               stored (usually named label_state_align).
        :param dir_out:        Main directory where the labels and
                               normalisation parameters are saved to.
                               If None, labels are not saved.
        :param file_id_list:   Name of the file containing the ids.
                               Normalisation parameters are saved using
                               this name to differentiate parameters
                               between subsets.
        :param id_list:        The list of utterances to process. Should
                               have the form uttId1 \\n uttId2 \\n ...
                               \\n uttIdN. If None, all file in dir_in
                               are used.
        :param return_dict:    If true, returns an OrderedDict of all
                               samples as first output.
        :return:               Returns two normalisation parameters as
                               tuple. If return_dict is True it returns
                               all processed labels in an OrderedDict
                               followed by the two normalisation
                               parameters.
        """

        # Fill file_id_list by .wav files in dir_in if not given and set
        # an appropriate file_id_list_name.
        if id_list is None:
            id_list = list()
            filenames = glob.glob(os.path.join(dir_in, "*.wav"))
            for filename in filenames:
                id_list.append(os.path.splitext(os.path.basename(filename))[0])
            file_id_list_name = "all"
        else:
            file_id_list_name = os.path.splitext(os.path.basename(file_id_list))[0]
            id_list = ['{}'.format(os.path.basename(element))
                       for element in id_list]  # Ignore full path.

        if dir_out is not None:
            makedirs_safe(dir_out)

        if return_dict:
            label_dict = OrderedDict()

        norm_params_ext_dur = MeanStdDevExtractor()

        logging.info("Extract phoneme durations for " + "[{0}]".format(
            ", ".join(str(i) for i in id_list)))
        for file_name in id_list:
            logging.debug("Extract phoneme durations from " + file_name)

            if label_type == "full_state_align":
                file_path = os.path.join(dir_in, file_name + ".lab")
                dur = PhonemeDurationLabelGen._get_full_state_align_dur(
                    file_path,
                    PhonemeDurationLabelGen.min_phoneme_length,
                    PhonemeDurationLabelGen.num_states)
            elif label_type == "mfa":
                file_path = os.path.join(dir_in, file_name + ".TextGrid")
                dur = PhonemeDurationLabelGen._get_mfa_dur(
                    file_path,
                    PhonemeDurationLabelGen.frame_length_sec)
            else:
                raise NotImplementedError("Unknown label type {}.".format(
                    label_type))

            if return_dict:
                label_dict[file_name] = dur
            if dir_out is not None:
                file_path = os.path.join(dir_out, file_name)
                LabelGen._save_to_npz(file_path, dur, "dur")

            norm_params_ext_dur.add_sample(dur)

        norm_params_ext_dur.save(os.path.join(dir_out, file_id_list_name))

        norm_first, norm_second = norm_params_ext_dur.get_params()

        if return_dict:
            return label_dict, norm_first, norm_second
        else:
            return norm_first, norm_second

    @staticmethod
    def _get_full_state_align_dur(file_path: os.PathLike, min_length: int,
                                  num_states: int):
        with open(file_path, 'r') as f:
            htk_labels = [line.rstrip('\n').split()[:2] for line in f]
            timings = np.array(htk_labels, dtype=np.float32) / min_length
            dur = timings[:, 1] - timings[:, 0]
            dur = dur.reshape(-1, num_states).astype(np.float32)
        return dur

    @staticmethod
    def _get_mfa_dur(file_path: os.PathLike, frame_length_sec: float):
        import textgrid
        tg = textgrid.TextGrid.fromFile(file_path)

        intervals = [tier for tier in tg if tier.name == "phones"][0]
        dur = [(phoneme.maxTime - phoneme.minTime) / frame_length_sec
               for phoneme in intervals]
        return np.array(dur, dtype=np.float32)


def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "-l", "--dir_labels",
        help="Directory containing the label (HTK full labels, *.lab) files.",
        type=str, dest="dir_labels", required=True)
    parser.add_argument(
        "-t", "--label_type",
        help="Type of the labels saved in the dir_labels directory.",
        choices=("full_state_align", "mfa"), type=str, dest="label_type",
        default="full_state_align", required=False)
    parser.add_argument(
        "-i", "--file_id_list_path",
        help="Path to text file to read the ids of the files to process. "
             "Default uses all .wav files in the given audio_dir.",
        type=str, dest="file_id_list_path", default=None)
    parser.add_argument(
        "-o", "--dir_out", help="Output directory to store the labels. Within "
        "the output directory folders for each feature will be created.",
        type=str, dest="dir_out", required=True)

    args = parser.parse_args()

    label_type = args.label_type
    dir_out = os.path.abspath(args.dir_out)
    dir_labels = os.path.abspath(args.dir_labels)

    # Read ids and select an appropriate file_id_list_name,
    # used to identify normalisation parameters of different subsets.
    if args.file_id_list_path is not None:
        file_id_list_path = os.path.abspath(args.file_id_list_path)

        with open(file_id_list_path) as f:
            id_list = f.readlines()

        # Trim entries in-place.
        id_list[:] = [s.strip(' \t\n\r') for s in id_list]
        file_id_list_name = os.path.splitext(
            os.path.basename(file_id_list_path))[0]
    else:
        id_list = None
        file_id_list_name = "all"

    PhonemeDurationLabelGen.gen_data(
        dir_in=dir_labels,
        dir_out=dir_out,
        file_id_list=args.file_id_list_path,
        id_list=id_list,
        label_type=label_type,
        return_dict=False)

    sys.exit(0)


if __name__ == "__main__":
    main()
