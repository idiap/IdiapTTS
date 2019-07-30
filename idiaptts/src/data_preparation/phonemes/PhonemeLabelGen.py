#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#

"""Module description:
   Create question labels for HTK labels.
"""

import argparse
import glob
# System imports.
import logging
import math
import os
import sys
from collections import OrderedDict
import numpy as np

# Third-party imports.

# Local source tree imports.
from idiaptts.misc.normalisation.MinMaxExtractor import MinMaxExtractor
from idiaptts.src.data_preparation.LabelGen import LabelGen
from idiaptts.src.data_preparation.questions.label_normalisation import HTSLabelNormalisation
from idiaptts.misc.utils import makedirs_safe, file_len


class PhonemeLabelGen(LabelGen):
    """
    A class to load phonemes from full HTK labels. Durations in the labels are ignored so they do not need to be
    aligned. The labels have to be generated before. This class does not provide any generation functionality.

    """

    ext_phonemes = ".lab"
    logger = logging.getLogger(__name__)

    def __init__(self, dir_labels, file_symbol_dict, label_type="HTK full", add_EOF=False):

        # Attributes.
        self.dir_labels = dir_labels
        self.symbol_dict = self._get_symbol_dict(file_path_full=file_symbol_dict)
        self.num_symbols = len(self.symbol_dict)
        self.label_type = label_type
        self.add_EOF = add_EOF
        self.norm_params = None

    def __getitem__(self, id_name):
        """Return the preprocessed sample with the given id_name."""
        sample = self.load_sample(id_name, self.dir_labels, self.symbol_dict, self.label_type)
        sample = self.preprocess_sample(sample)

        return sample

    def _get_symbol_dict(self, file_path_full):
        with open(file_path_full) as f:
            symbols = f.read()
        symbols = symbols.split()

        symbol_dict = dict()
        for id, symbol in enumerate(symbols):
            symbol_dict[symbol] = id
        symbol_dict['EOF'] = len(symbol_dict)

        return symbol_dict

    def _symbol_to_id(self, symbol):
        raise NotImplementedError()

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

    def preprocess_sample(self, sample, norm_params=None):

        if self.add_EOF:
            extended_sample = np.empty((sample.shape[0] + 1, *sample.shape[1:]), dtype=sample.dtype)  # Create one more for EOF symbol.
            extended_sample[:len(sample)] = sample
            extended_sample[-1] = self.symbol_dict['EOF']  # Add EOF symbol.
        return sample

    def postprocess_sample(self, sample, norm_params=None):
        if self.add_EOF:
            sample = sample[:-1]  # Remove EOF symbol.
        return sample

    @staticmethod
    def load_sample(id_name, dir_out, symbol_dict, label_type="HTK full"):
        label_file = os.path.join(dir_out, id_name + PhonemeLabelGen.ext_phonemes)
        with open(label_file) as f:
            if label_type == "HTK full":  # Input are HTK full labels.
                symbols = list()
                for line in f.read().split('\n')[::5]:  # Read only one line per state.
                    if len(line) > 0:  # Skip empty lines.
                        line_split = line.split()
                        full_label = line_split[2]
                        symbol = full_label.split('-')[1].split('+')[0]
                        symbols.append(symbol)
            elif label_type == "mono_no_align":  # Input are files with one phoneme per line (mono_no_align).
                symbols = f.read()
                symbols = symbols.split()
            else:
                raise NotImplementedError("Unknown label type {} while loading {}.".format(label_type, label_file))

        ids = np.zeros((len(symbols), 1), dtype=np.float32)
        # Convert symbols to ids.
        for index, symbol in enumerate(symbols):
            ids[index] = symbol_dict[symbol]

        return ids

    def gen_data(self, dir_in, dir_out=None, file_id_list=None, id_list=None, return_dict=False):
        raise NotImplementedError("The generation of monophones is not implemented."
                                  "Please use festival to create Monophones, or be the first to implement it!")


def main():
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-l", "--dir_labels", help="Directory containing the label (HTK full labels, *.lab) files.",
                        type=str, dest="dir_labels", required=True)
    parser.add_argument("-d", "--file_dict", help="Full path to file that contains the dictionary of used Monophones.",
                        type=str, dest="file_phonemes", required=True)
    # parser.add_argument("-n", "--num_questions", help="Number of questions in the question file.",
    #                     type=int, dest="num_questions", default=425)
    parser.add_argument("-i", "--file_id_list_path",
                        help="Path to text file to read the ids of the files to process.\
                              Default uses all .lab files in the given label_dir.",
                        type=str, dest="file_id_list_path", default=None)
    parser.add_argument("-o", "--dir_out", help="Output directory to store the labels.",
                        type=str, dest="dir_out", required=True)
    parser.add_argument("-t", "--label_type",
                        help="Type of the labels saved in the dir_labels directory.",
                        choices=("HTK full", "mono_no_align"),
                        type=str, dest="label_type", default="HTK full", required=False)

    # Parse arguments
    args = parser.parse_args()

    dir_labels = os.path.abspath(args.dir_labels)
    file_phonemes = os.path.abspath(args.file_phonemes)
    if args.file_id_list_path is not None:
        file_id_list_path = os.path.abspath(args.file_id_list_path)
        # Read which files to process.
        with open(file_id_list_path) as f:
            id_list = f.readlines()
        # Trim entries in-place.question
        id_list[:] = [s.strip(' \t\n\r') for s in id_list]
    else:
        id_list = None

    dir_out = os.path.abspath(args.dir_out)

    # Execute main functionality.
    # PhonemeLabelGen.gen_data(dir_labels, file_phonemes, dir_out=dir_out, file_id_list=args.file_id_list_path, id_list=id_list, return_dict=False)

    # DEBUG
    phoneme_gen = PhonemeLabelGen(dir_labels, file_phonemes, args.label_type)
    test_label = phoneme_gen["LJ001-0007"]
    print(test_label)

    sys.exit(0)


if __name__ == "__main__":
    main()
