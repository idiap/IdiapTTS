#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#

# System imports.
import argparse
import logging
import os
from typing import  Dict, List, Union

# Third-party imports.
import numpy as np

# Local source tree imports.


class MinMaxExtractor:
    """
    A class to collect minimum and maximum of data points.
    Calling it as a script is used to combine min and max of different
    subsets.
    """
    logger = logging.getLogger(__name__)

    # Constants.
    file_name_appendix = "min-max"

    def __init__(self):
        self.combined_min = float("inf")
        self.combined_max = -float("inf")

    def _normalise(self, feature, min_, max_):
        range_ = max_ - min_
        MinMaxExtractor._fix_range_inplace(range_)
        return (feature - min_) / range_
        # return (0.01 + (feature - min_) / range_) / 0.99
        # TODO: This should be 0.01 + (feature - min_) / range_ * 0.98,
        #       change denormalise as well then.

    def _denormalise(self, feature, min_, max_):
        range_ = max_ - min_
        MinMaxExtractor._fix_range_inplace(range_)
        return feature * range_ + min_
        return (feature * 0.99 - 0.01) * range_ + min_

    @staticmethod
    def _fix_range_inplace(range_):
        """Prevent division by negative values or zero."""

        is_zero = range_ == 0
        if any(is_zero):
            range_[is_zero] = 1

        is_neg = range_ < 0
        if any(is_neg):
            indices = np.arange((len(range_)))[is_neg]
            range_values = range_[is_neg]
            idx_range_array = np.stack((indices, range_values), axis=1)
            # TODO: Should this throw a ValueError?
            logging.warning("Found negative range(s), setting them to 1 for "
                            "feature(s)\n[index\tvalue]\n{}".format(
                                idx_range_array))
            range_[is_neg] = 1

    def add_sample(self, sample):

        self.combined_min = np.minimum(self.combined_min, sample.min(axis=0))
        self.combined_max = np.maximum(self.combined_max, sample.max(axis=0))

    def get_params(self):
        return self.combined_min, self.combined_max

    def save(self, filename, datatype=np.float64):

        if filename is not None and os.path.basename(filename) != "":
            filename += "-"

        samples_min, samples_max = self.get_params()
        self._save(filename=filename + self.file_name_appendix,
                   stats={"min": samples_min, "max": samples_max},
                   datatype=datatype)

    @staticmethod
    def _save(filename: Union[str, os.PathLike], stats: Dict[str, np.ndarray],
              datatype: str) -> None:
        if datatype is np.str:
            stats_concatenated = np.stack(list(stats.values()), axis=0)
            np.savetxt(filename + ".txt", stats_concatenated)
        elif datatype is np.float32 or datatype is np.float64:
            np.savez(filename, **stats)
        else:
            logging.error("Unknown datatype: {}. Please choose one of "
                          "[numpy.float32, numpy.float64, numpy.str]."
                          .format(datatype.__name__))

    @staticmethod
    def load(file_path, datatype=np.float64):
        if datatype is np.str:
            min_max = np.loadtxt(file_path, dtype=np.float32).reshape((2, -1))
            samples_min, samples_max = np.split(min_max, min_max.shape[0],
                                                axis=0)
        elif datatype is np.float32 or datatype is np.float64:
            if file_path.endswith(".bin"):
                with open(file_path, 'rb') as f:
                    min_max = np.fromfile(f, dtype=datatype).reshape((2, -1))
                    samples_min, samples_max = np.split(min_max,
                                                        min_max.shape[0],
                                                        axis=0)
            else:
                saved_archive = np.load(file_path)
                samples_min = saved_archive["min"]
                samples_max = saved_archive["max"]
        else:
            logging.error("Unknown datatype: {}. Please choose one of "
                          "[numpy.float32, numpy.float64, numpy.str]."
                          .format(datatype.__name__))
            return None

        return samples_min.squeeze(), samples_max.squeeze()

    @staticmethod
    def combine_min_max(file_list, dir_out=None, datatype=np.float64,
                        save_txt=False):
        """
        Computes the combined min and max from all min and max of the
        subgroups.

        :param file_list:   List where each entry contains the filename
                            (with path) of the files to merge.
        :param dir_out:     If given, saves the combined parameters to
                            that directory as .npz.
        :param datatype:    Format in which the numpy arrays are stored
                            in the files.
        :param save_txt:    Also saves the parameters in a .txt file.
        :return:            Returns the combined normalisation parameters.
        """
        samples_min = float("inf")
        samples_max = -float("inf")

        for file in file_list:
            current_min, current_max = MinMaxExtractor.load(file,
                                                            datatype=datatype)

            samples_min = np.minimum(samples_min, current_min.squeeze())
            samples_max = np.maximum(samples_max, current_max.squeeze())

        if dir_out is not None:
            filename = os.path.join(dir_out,
                                    MinMaxExtractor.file_name_appendix)
            stats_dict = {"min": samples_min, "max": samples_max}
            MinMaxExtractor._save(filename, stats_dict, datatype=datatype)

            if save_txt:
                MinMaxExtractor._save(filename, stats_dict, datatype=np.str)

        return samples_min, samples_max


def main():
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-f", "--file_list",
                        help="List of files containing the parameters.",
                        nargs='+', dest="file_list", required=True)
    parser.add_argument("-o", "--dir_out",
                        help="Directory to save the combined parameters.",
                        type=str, dest="dir_out", required=False, default=None)
    parser.add_argument("-t", "--dtype",
                        help="Format in which numpy array is saved.",
                        default="float64", dest="dtype",
                        choices=["float32", "float64"], required=False)

    args = parser.parse_args()

    if args.dtype == "float32":
        datatype = np.float32
    else:
        datatype = np.float64

    MinMaxExtractor.combine_min_max(args.file_list, args.dir_out, datatype)


if __name__ == "__main__":
    main()
