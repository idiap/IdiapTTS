#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#

"""Module description:
   Create phoneme index vectors for different label files.
"""

# System imports.
import argparse
from functools import partial
import logging
import os
import sys
from typing import Dict, Union

# Third-party imports.
import numpy as np

# Local source tree imports.
from idiaptts.src.data_preparation.LabelGen import LabelGen
from idiaptts.src.data_preparation.NpzDataReader import NpzDataReader


class PhonemeLabelGen(NpzDataReader, LabelGen):
    """
    A class to load phonemes from different labels. Durations in the
    labels are ignored, if present, so they do not need to be aligned.
    The labels have to be generated before. This class does not provide
    any generation functionality.
    """

    ext_phonemes = ".lab"
    eof_symbol = 'EOF'
    silent_symbol = 'sil'
    logger = logging.getLogger(__name__)

    class Config(NpzDataReader.Config):

        def __init__(self,
                     directory,
                     *args,
                     file_symbol_dict: Union[os.PathLike, str] = None,
                     symbol_dict: Dict = None,
                     norm_type: str = NpzDataReader.Config.NormType.NONE,
                     pad_mode: str = 'constant',
                     label_type: str = "HTK full",
                     add_EOF: bool = False,
                     one_hot: bool = False,
                     **kwargs) -> None:
            super().__init__(*args, directory=directory, norm_type=norm_type,
                             pad_mode=pad_mode, **kwargs)

            self.file_symbol_dict = file_symbol_dict
            self.symbol_dict = symbol_dict
            self.label_type = label_type
            self.add_EOF = add_EOF
            self.one_hot = one_hot

        def create_reader(self):
            reader = PhonemeLabelGen(self)

            reader.get_normalisation_params()

            return reader

    def __init__(self, *args, **kwargs):

        if len(args) == 1 \
                and isinstance(args[0], PhonemeLabelGen.Config):
            config = args[0]

            file_symbol_dict = config.file_symbol_dict
            symbol_dict = config.symbol_dict
            self.label_type = config.label_type
            self.add_EOF = config.add_EOF
            self.one_hot = config.one_hot

            self.legacy_getitem = False
        else:
            # LEGACY support
            if len(args) > 0:
                dir_labels = args[0]
            else:
                dir_labels = kwargs["dir_labels"]
            if len(args) > 1:
                file_symbol_dict = args[1]
            else:
                file_symbol_dict = kwargs["file_symbol_dict"]
            if len(args) > 2:
                self.label_type = args[2]
            else:
                self.label_type = kwargs.get("label_type", "HTK full")
            if len(args) > 3:
                self.add_EOF = args[3]
            else:
                self.add_EOF = kwargs.get("add_EOF", False)
            if len(args) > 4:
                self.one_hot = args[4]
            else:
                self.one_hot = kwargs.get("one_hot", False)

            symbol_dict = None  # Was not support in legacy code.

            config = PhonemeLabelGen.Config(
                name="phonemes",
                directory=dir_labels,
                file_symbol_dict=file_symbol_dict,
                norm_type=NpzDataReader.Config.NormType.NONE)

            self.legacy_getitem = True

        preprocessing_functions = list()
        if self.add_EOF:
            preprocessing_functions.append(self._add_eof_symbol)
        if self.one_hot:
            preprocessing_functions.append(self._convert_to_one_hot)
        if config.preprocessing_fn is not None:
            preprocessing_functions.append(config.preprocessing_fn)
        if len(preprocessing_functions) > 0:
            config.preprocessing_fn = partial(
                self._preprocess,
                preprocessing_fn=config.preprocessing_fn)

        postprocessing_functions = list()
        if self.add_EOF:
            postprocessing_functions.append(self._remove_eof_symbol)
        if config.postprocessing_fn is not None:
            postprocessing_functions.append(config.postprocessing_fn)
        if len(postprocessing_functions) > 0:
            config.postprocessing_fn = partial(
                self._postprocess,
                postprocessing_fn=config.postprocessing_fn)
        super().__init__(config)

        if symbol_dict is not None:
            self.symbol_dict = symbol_dict
        else:
            if file_symbol_dict is not None:
                self.symbol_dict = self.get_symbol_dict(file_symbol_dict)
            # else:
            #     self.logger.warning("No symbol dict defined.")
        self.num_symbols = len(self.symbol_dict)
        self.symbol_one_hot = np.eye(len(self.symbol_dict), dtype=np.float32)

    def _preprocess(self, sample, preprocessing_fn):
        if self.add_EOF:
            sample = self._add_eof_symbol(sample)
        if self.one_hot:
            sample = self._convert_to_one_hot(sample)
        if preprocessing_fn is not None:
            sample = preprocessing_fn(sample)
        return sample

    def _postprocess(self, sample, postprocessing_fn):
        if self.postprocessing_fn is not None:
            sample = postprocessing_fn(sample)
        if self.add_EOF:
            sample = self._remove_eof_symbol(sample)
        return sample

    def __getitem__(self, id_name):
        """Return the preprocessed sample with the given id_name."""
        sample_dict = super().__getitem__(id_name)
        if self.legacy_getitem:
            # LEGACY support
            return sample_dict[self.output_names[0]]
        else:
            return sample_dict

    @staticmethod
    def get_symbol_dict(file_path_full):
        with open(file_path_full) as f:
            symbols = f.read()
        symbols = symbols.split()

        symbol_dict = dict()
        for id, symbol in enumerate(symbols):
            symbol_dict[symbol] = id
        symbol_dict[PhonemeLabelGen.eof_symbol] = len(symbol_dict)

        return symbol_dict

    def _symbol_to_id(self, symbol):
        raise NotImplementedError()

    def _add_eof_symbol(self, sample):
        extended_sample = np.empty((sample.shape[0] + 1, *sample.shape[1:]),
                                   dtype=sample.dtype)
        extended_sample[:len(sample)] = sample
        extended_sample[-1] = self.symbol_dict[PhonemeLabelGen.eof_symbol]

        return extended_sample

    def _convert_to_one_hot(self, sample):
        return np.squeeze(self.symbol_one_hot[sample.reshape(-1)])

    def _remove_eof_symbol(self, sample):
        return sample[:-1]

    @staticmethod
    def load_sample(id_name, dir_out, symbol_dict, label_type="HTK full"):
        datareader = PhonemeLabelGen.Config(
            name="phonemes",
            directory=dir_out,
            symbol_dict=symbol_dict,
            label_type=label_type
        ).create_reader()

        return datareader.load(id_name)

    def load(self, id_name: str):
        logging.debug("Load phonemes for " + id_name)
        try:
            symbols = super().load(id_name)

        except FileNotFoundError:
            # LEGACY support
            id_name = os.path.splitext(os.path.basename(id_name))[0]
            file_path = os.path.join(self.directory[0], id_name)

            # Input is HTK full state aligned.
            if self.label_type == "full_state_align":
                symbols = PhonemeLabelGen._load_htk_state_align(file_path)

            # Input are HTK full labels.
            elif self.label_type == "HTK full":
                symbols = PhonemeLabelGen._load_htk_full(file_path)

            # Input are files with one phoneme per line (mono_no_align).
            elif self.label_type == "mono_no_align":
                with open(file_path + ".lab") as f:
                    symbols = f.read()
                    symbols = symbols.split()
            elif self.label_type == "mfa":
                symbols = PhonemeLabelGen._load_mfa(file_path)
            else:
                raise NotImplementedError("Unknown label type {} while loading"
                                          " {}.".format(self.label_type,
                                                        file_path))

        ids = np.zeros((len(symbols), 1), dtype=np.long)
        # Convert symbols to ids.
        for index, symbol in enumerate(symbols):
            ids[index] = self.symbol_dict[symbol]

        return ids

    @staticmethod
    def _load_htk_state_align(file_path):
        """This method expects each phoneme to have 5 states."""
        with open(file_path + ".lab") as f:
            assert f.readline().strip('\n')[-3:] == "[2]",\
                "Labels do not seem to contain state information."\
                " Are you using the correct label_type?"
            f.seek(0)

            symbols = list()

            for line in f.read().split('\n')[::5]:  # Read only one line per state.
                if len(line) == 0:  # Skip empty lines.
                    continue
                symbols.append(PhonemeLabelGen._read_symbol_from_htk_full(line))

        return symbols

    @staticmethod
    def _load_htk_full(file_path):
        """This method expects each phoneme to have only one state."""
        with open(file_path + ".lab") as f:
            assert f.readline().strip('\n')[-3:] != "[2]",\
                "Labels do seem to contain state information. Are "\
                "you using the correct label_type?"
            f.seek(0)

            symbols = list()
            for line in f.read().split('\n'):  # Read only one line per state.
                if len(line) == 0:  # Skip empty lines.
                    continue
                symbols.append(PhonemeLabelGen._read_symbol_from_htk_full(line))

        return symbols

    @staticmethod
    def _load_mfa(file_path):
        import textgrid
        tg = textgrid.TextGrid.fromFile(file_path + ".TextGrid")

        intervals = [tier for tier in tg if tier.name == "phones"][0]
        phonemes = list()
        for interval in intervals:
            if interval.mark != '':
                phonemes.append(interval.mark)
            else:
                # Since 2.0.0a22 MFA does not use sil or sp phones but leaves
                # the mark empty: https://montreal-forced-aligner.readthedocs.io/en/latest/changelog.html?highlight=SIL
                phonemes.append(PhonemeLabelGen.silent_symbol)
        return phonemes

    @staticmethod
    def _read_symbol_from_htk_full(line):
        line_split = line.split()
        full_label = line_split[2]
        symbol = full_label.split('-')[1].split('+')[0]
        return symbol

    def gen_data(self, dir_in, dir_out=None, file_id_list=None, id_list=None,
                 return_dict=False):
        raise NotImplementedError("The generation of monophones is not "
                                  "implemented. Please use a tool to "
                                  "create Monophones, or be the first "
                                  "to implement it!")


# def main():
#     logging.basicConfig(level=logging.DEBUG)

#     parser = argparse.ArgumentParser(
#         description=__doc__,
#         formatter_class=argparse.RawDescriptionHelpFormatter)
#     parser.add_argument("-l", "--dir_labels",
#                         help="Directory containing the label (HTK full "
#                         "labels, *.lab) files.", type=str, dest="dir_labels",
#                         required=True)
#     parser.add_argument("-d", "--file_dict",
#                         help="Full path to file that contains the "
#                         "dictionary of used Monophones.", type=str,
#                         dest="file_phonemes", required=True)
#     parser.add_argument("-t", "--label_type",
#                         help="Type of the labels saved in the "
#                         "dir_labels directory.",
#                         choices=("HTK full", "mono_no_align",
#                                  "full_state_align", "mfa"),
#                         type=str, dest="label_type", default="HTK full",
#                         required=False)
#     parser.add_argument("--add_eof", help="Add EOF symbol in every utterance.",
#                         dest="add_eof", action='store_true', default=False)
#     parser.add_argument("--one_hot", help="Convert to one hot vectors.",
#                         dest="one_hot", action='store_true', default=False)

#     args = parser.parse_args()

#     dir_labels = os.path.abspath(args.dir_labels)
#     file_phonemes = os.path.abspath(args.file_phonemes)

#     # DEBUG
#     phoneme_gen = PhonemeLabelGen(dir_labels, file_phonemes, args.label_type,
#                                   args.add_eof, args.one_hot)
#     test_label = phoneme_gen["LJ001-0007"]
#     print(test_label)

#     sys.exit(0)


# if __name__ == "__main__":
#     main()
