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
import struct
from typing import  Dict, List, Union

# Third-party imports.
import numpy as np

# Local source tree imports.


class MeanStdDevExtractor(object):
    """
    A class to collect mean and standard deviation of data in an online
    manner. Calling it as a script is used to combine mean and standard
    deviation of different subsets.
    """
    logger = logging.getLogger(__name__)

    file_name_stats = "stats"
    file_name_appendix = "mean-std_dev"

    def __init__(self):
        self.sum_length = 0
        self.sum_frames = 0
        self.sum_squared_frames = 0

    def _normalise(self, feature, mean, std_dev):
        return (feature - mean) / std_dev

    def _denormalise(self, feature, mean, std_dev):
        return feature * std_dev + mean

    def add_sample(self, sample):
        assert sample is not None, "Sample cannot be None."
        self.sum_length += len(sample)
        self.sum_frames += np.sum(sample, axis=0)
        self.sum_squared_frames += np.sum(sample**2, axis=0)

    def get_params(self):
        mean = self.sum_frames / self.sum_length
        std_dev = np.sqrt(self.sum_squared_frames / self.sum_length - mean**2)

        return np.atleast_1d(mean), np.atleast_1d(std_dev)

    def save(self, filename, datatype=np.float64):
        self.save_stats(filename, datatype)
        self.save_mean_std_dev(filename, datatype)

    def save_stats(self, filename, datatype=np.float64):

        if filename is not None and os.path.basename(filename) != "":
            filename += "-"

        self._save(filename=filename + self.file_name_stats,
                   sum_length=self.sum_length,
                   stats={"sum_frames": self.sum_frames,
                          "sum_squared_frames": self.sum_squared_frames},
                   datatype=datatype)

    def save_mean_std_dev(self, filename, datatype=np.float64):
        if filename is not None and os.path.basename(filename) != "":
            filename += "-"

        mean, std_dev = self.get_params()
        self._save(filename=filename + self.file_name_appendix,
                   sum_length=self.sum_length,
                   stats={"mean": mean, "std_dev": std_dev},
                   datatype=datatype)

    @staticmethod
    def _save(filename: Union[str, os.PathLike], sum_length: int,
              stats: Dict[str, np.ndarray], datatype: str) -> None:
        if datatype is np.str:
            stats_concatenated = np.concatenate(list(stats.values()),
                                                axis=0)
            np.savetxt(filename + ".txt", stats_concatenated,
                       header=str(sum_length))

        elif datatype is np.float32 or datatype is np.float64:
            stats = {k: np.atleast_1d(v).astype(datatype, copy=False)
                     for k, v in stats.items()}
            stats["sum_length"] = np.array(sum_length, dtype=np.int)

            np.savez(filename, **stats)
        else:
            logging.error("Unknown datatype: {}. Please choose one of "
                          "[numpy.float32, numpy.float64, numpy.str]."
                          .format(datatype.__name__))

    @staticmethod
    def load_stats(file_path, datatype=np.float64):
        if datatype is np.str:
            with open(file_path, 'r') as f:
                labels_len = int(f.readline())
                stats = np.loadtxt(f, dtype=np.float64)
                sum_frames, sum_squared_frames = np.split(stats,
                                                          stats.shape[0],
                                                          axis=0)
        elif datatype is np.float32 or datatype is np.float64:
            saved_archive = np.load(file_path)
            labels_len = saved_archive["sum_length"]
            sum_frames = saved_archive["sum_frames"]
            sum_squared_frames = saved_archive["sum_squared_frames"]
        else:
            logging.error("Unknown datatype: {}. Please choose one of "
                          "[numpy.float32, numpy.float64, numpy.str]."
                          .format(datatype.__name__))
            return None

        return sum_frames, sum_squared_frames, labels_len

    @staticmethod
    def load(file_path, datatype=np.float64):
        if datatype is np.str:
            with open(file_path, 'r') as f:
                f.readline()  # Skip first line.
                mean_std_dev = np.loadtxt(f, dtype=np.float32)
                mean, std_dev = np.split(mean_std_dev, mean_std_dev.shape[0],
                                         axis=0)
        elif datatype is np.float32 or datatype is np.float64:
            if file_path.endswith(".bin"):  # LEGACY support
                with open(file_path, 'rb') as f:
                    _ = struct.unpack("i", f.read(4))[0]  # Read over sum_length.
                    mean_std_dev = np.fromfile(f, dtype=datatype).reshape(
                        (2, -1))
                mean, std_dev = np.split(mean_std_dev, mean_std_dev.shape[0],
                                         axis=0)
            else:
                saved_archive = np.load(file_path)
                mean = saved_archive["mean"]
                std_dev = saved_archive["std_dev"]
        else:
            logging.error("Unknown datatype: {}. Please choose one of "
                          "[numpy.float32, numpy.float64, numpy.str]."
                          .format(datatype.__name__))
            return None

        return (mean.astype(np.float32, copy=False),
                std_dev.astype(np.float32, copy=False))

    @staticmethod
    def load_mean_std_dev_from_stats(file_path, datatype=np.float64):
        sum_frames, sum_squared_frames, sum_length = \
            MeanStdDevExtractor.load_stats(file_path, datatype)

        mean = sum_frames / sum_length
        std_dev = np.sqrt(sum_squared_frames / sum_length - mean ** 2)

        return (mean.astype(np.float32, copy=False),
                std_dev.astype(np.float32, copy=False))

    @staticmethod
    def combine_stats(file_list: List[Union[str, os.PathLike]],
                      dir_out: Union[str, os.PathLike] = None,
                      datatype: str = np.float64,
                      save_txt: bool = False):
        """
        Combines the stats of different subsets.

        :param file_list:   List where each entry contains the
                            filename (with path) of the files to merge.
        :param dir_out:     If given, saves the combined parameters to
                            that directory as .npz.
        :param datatype:    Format in which the numpy arrays are stored
                            in the files.
        :param save_txt:    Also saves the parameters in a .txt file.
        :return:            Returns the combined normalisation parameters.
        """

        sum_length = 0
        sum_frames = 0
        sum_squared_frames = 0

        for file in file_list:
            current_sum_frames, current_sum_squared_frames, labels_len = \
                MeanStdDevExtractor.load_stats(file, datatype=datatype)

            sum_length += labels_len
            sum_frames += current_sum_frames
            sum_squared_frames += current_sum_squared_frames

        if dir_out is not None:
            filename = os.path.join(dir_out,
                                    MeanStdDevExtractor.file_name_stats)
            stats_dict = {"sum_frames": sum_frames,
                          "sum_squared_frames": sum_squared_frames}
            MeanStdDevExtractor._save(filename, sum_length, stats_dict,
                                      datatype=np.float32)

            if save_txt:
                MeanStdDevExtractor._save(filename, sum_length, stats_dict,
                                          datatype=np.str)

        return sum_length, sum_frames, sum_squared_frames

    @staticmethod
    def combine_mean_std(file_list: List[Union[str, os.PathLike]],
                         dir_out: Union[str, os.PathLike] = None,
                         datatype: str = np.float64,
                         save_txt: bool = True):
        """
        Combines the mean and standard deviation of different subsets.
        Under the hood it calls combine_stats and computes the mean and
        standard deviation from the combined statistics.

        :param file_list:   List where each entry contains the filename
                            (with path) of the files to merge.
        :param dir_out:     If given, saves the combined parameters to
                            that directory as .npz.
        :param datatype:    Format in which the numpy arrays are stored
                            in the files.
        :param save_txt:    Also saves the parameters in a .txt file.
        :return:            Returns the combined normalisation parameters.
        """

        sum_length, sum_frames, sum_squared_frames = \
            MeanStdDevExtractor.combine_stats(file_list, dir_out=dir_out,
                                              datatype=datatype)

        mean = sum_frames / sum_length
        variance = sum_squared_frames / sum_length - mean ** 2
        negative_entries = (variance < 0)[0]
        if negative_entries.any():
            logging.warning("Encountered negative variance for indices {} ({})"
                            "when combining statistics of {}. Setting those "
                            "elements to 0 instead.".format(
                                np.arange(variance.shape[1])[negative_entries],
                                variance[:, negative_entries],
                                file_list))
            variance[:, negative_entries] = 0.0
        std_dev = np.sqrt(variance)

        if dir_out is not None:
            filename = os.path.join(dir_out,
                                    MeanStdDevExtractor.file_name_appendix)
            stats_dict = {"mean": mean,
                          "std_dev": std_dev}
            MeanStdDevExtractor._save(filename, sum_length, stats_dict,
                                      datatype=datatype)

            if save_txt:
                MeanStdDevExtractor._save(filename, sum_length, stats_dict,
                                          datatype=np.str)

        return mean, std_dev


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

    MeanStdDevExtractor.combine_mean_std(args.file_list, args.dir_out,
                                         datatype)


if __name__ == "__main__":
    main()
