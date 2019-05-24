#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#

"""Module description:
   Create world lf0 and vuv feature labels for .wav files.
   By default removes the phrase curve from LF0 when loading it.
"""

# System imports.
import numpy as np
import os

# Local source tree imports.
from misc.normalisation.MeanStdDevExtractor import MeanStdDevExtractor
from src.data_preparation.world.LF0LabelGen import LF0LabelGen


class FlatLF0LabelGen(LF0LabelGen):
    """
    Load LF0 and (by default) remove phrase curve from it.
    The atom amplitude normalisation is used to normalise the LF0 curve.
    """
    ext_phrase = ".phrase"

    def __init__(self, dir_lf0_labels, dir_phrase_labels, remove_phrase=True):
        super().__init__(dir_labels=dir_lf0_labels, add_deltas=False)
        self.dir_phrase = dir_phrase_labels
        self.remove_phrase = remove_phrase

    def get_phrase_curve(self, id_name):
        return np.fromfile(os.path.join(self.dir_phrase, id_name + self.ext_phrase), dtype=np.float32)

    def __getitem__(self, id_name):
        """Return the preprocessed sample with the given id_name."""
        sample = self.load_sample(id_name, self.dir_labels)
        if self.remove_phrase:
            phrase_curve = self.get_phrase_curve(id_name)
            sample[:, 0] -= phrase_curve
        sample = self.preprocess_sample(sample)

        return sample

    def get_normalisation_params(self, dir_out, file_name=None):
        """
        Read the mean std_dev values from a file.
        Save them in self.norm_params.

        :param dir_out:       Directory containing the normalisation file, usually the atom directory.
        :param file_name:     Prefix of normalisation file.
                              Expects file to be named <file_name-><MeanStdDevExtractor.file_name_appendix>.bin
        :return:              Tuple of normalisation parameters (mean, std_dev).
        """

        full_file_name = (file_name + "-" if file_name is not None else "") + MeanStdDevExtractor.file_name_appendix + ".bin"

        # Use the same normalisation parameters for the LF0 curve without phrase curve
        # as for atoms. The phrase directory is the same as the atom directory.
        mean, std_dev = MeanStdDevExtractor.load(os.path.join(self.dir_phrase, full_file_name))
        mean, std_dev = mean[:, 0:1], std_dev[:, 0:1]  # Dimension of both is 1 x 2(atom amplitude, theta).

        # Manually set V/UV normalisation parameters and save the concatenated normalisation parameters locally.
        self.norm_params = np.concatenate((mean, np.zeros((1, 1))), axis=1),\
                           np.concatenate((std_dev, np.ones((1, 1))), axis=1)

        return self.norm_params
