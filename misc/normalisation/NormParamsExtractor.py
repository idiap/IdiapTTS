#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#

"""Module description.

"""

# System imports.
import logging
import numpy as np

# Third-party imports.

# Local source tree imports.
from misc.normalisation.MeanStdDevExtractor import MeanStdDevExtractor
from misc.normalisation.MinMaxExtractor import MinMaxExtractor


class NormParamsExtractor(object):
    """Base class for all normalization extractor classes.
    """
    logger = logging.getLogger(__name__)

    # Constants.
    file_name_min_max = "min-max.txt"
    file_name_stats = "stats"
    file_name_mean_std_dev = "mean-std_dev.txt"

    def __init__(self, mean_std_dev=False, min_max=False):
        self.normalisers = list()

        if mean_std_dev:
            self.normalisers.append(MeanStdDevExtractor())
        if min_max:
            self.normalisers.append(MinMaxExtractor())

    def add_sample(self, sample):
        for normaliser in self.normalisers:
            normaliser.add_sample(sample)

    def get_mean_std_dev(self):
        function_name = self.get_mean_std_dev.__name__
        for normaliser in self.normalisers:
            if callable(getattr(normaliser, function_name, None)):
                return normaliser.get_params()

        self.logger.error("No normaliser provides a " + function_name + " method."
                          "Use mean_std_dev=True in the constructor to provide one.")
        return None

    def get_min_max(self):
        function_name = self.get_min_max.__name__
        for normaliser in self.normalisers:
            if callable(getattr(normaliser, function_name, None)):
                return normaliser.get_params()

        self.logger.error("No normaliser provides a " + function_name + " method."
                          "Use min_max=True in the constructor to provide one.")
        return None

    def save(self, filename, datatype=np.float64):
        for normaliser in self.normalisers:
            normaliser.save(filename, datatype)

    def load(self, file_path, datatype=np.float64):
        results = list()
        for normaliser in self.normalisers:
            results.append(normaliser.load(file_path + "-" + normaliser.file_name_appendix, datatype))

        return results
