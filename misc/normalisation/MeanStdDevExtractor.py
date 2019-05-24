#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#

# System imports.
import os
import struct
import argparse
import logging
import numpy as np


# Third-party imports.

# Local source tree imports.


class MeanStdDevExtractor(object):
    """A class to collect mean and standard deviation of data in an online manner.
    Calling it as a script is used to combine mean and standard deviation of different subsets.
    """
    logger = logging.getLogger(__name__)

    # Constants.
    file_name_stats = "stats"
    file_name_appendix = "mean-std_dev"

    def __init__(self):
        self.sum_length = 0
        self.sum_frames = None
        self.sum_squared_frames = None

    def add_sample(self, sample):

        self.sum_length += len(sample)
        if self.sum_frames is None:
            self.sum_frames = np.sum(sample, axis=0)
            self.sum_squared_frames = np.sum(sample**2, axis=0)
        else:
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

        stats = np.concatenate((np.atleast_1d(self.sum_frames), np.atleast_1d(self.sum_squared_frames)))

        if datatype is np.str:
            np.savetxt(filename + "-" + self.file_name_stats + ".txt", stats, header=str(self.sum_length))
        elif datatype is np.float32 or datatype is np.float64:
            with open(filename + "-" + self.file_name_stats + ".bin", 'wb') as file:
                file.write(struct.pack("i", self.sum_length))
                np.array(stats, dtype=datatype).tofile(file)
        else:
            self.logger.error("Unknown datatype: " + datatype.__name__ + ". "
                              "Please choose one of {numpy.float32, numpy.float64, numpy.str}.")

    def save_mean_std_dev(self, filename, datatype=np.float64):
        mean_std_dev = np.concatenate((self.get_params()), axis=0)

        if datatype is np.str:
            np.savetxt(filename + "-" + self.file_name_appendix + ".txt", mean_std_dev, header=str(self.sum_length))
        elif datatype is np.float32 or datatype is np.float64:
            with open(filename + "-" + self.file_name_appendix + ".bin", 'wb') as file:
                file.write(struct.pack("i", self.sum_length))
                np.array(mean_std_dev, dtype=datatype).tofile(file)
        else:
            self.logger.error("Unknown datatype: " + datatype.__name__ + ". "
                              "Please choose one of {numpy.float32, numpy.float64, numpy.str}.")

    @staticmethod
    def load_stats(file_path, datatype=np.float64):
        if datatype is np.str:
            with open(file_path, 'r') as f:
                labels_len = int(f.readline())
                stats = np.loadtxt(f, dtype=np.float64)
        elif datatype is np.float32 or datatype is np.float64:
            with open(file_path, 'rb') as f:
                labels_len = struct.unpack("i", f.read(4))[0]
                stats = np.fromfile(f, dtype=datatype).reshape((2, -1))
        else:
            logging.error("Unknown datatype: " + datatype.__name__ + ". "
                          "Please choose one of {numpy.float32, numpy.float64, numpy.str}.")
            return None

        return stats, labels_len

    @staticmethod
    def load(file_path, datatype=np.float64):
        if datatype is np.str:
            with open(file_path, 'r') as f:
                labels_len = int(f.readline())
                mean_std_dev = np.loadtxt(f, dtype=np.float32)
        elif datatype is np.float32 or datatype is np.float64:
            with open(file_path, 'rb') as f:
                sum_length = struct.unpack("i", f.read(4))[0]
                mean_std_dev = np.fromfile(f, dtype=datatype).reshape((2, -1))
        else:
            logging.error("Unknown datatype: " + datatype.__name__ + ". "
                          "Please choose one of {numpy.float32, numpy.float64, numpy.str}.")
            return None

        mean, std_dev = np.split(mean_std_dev, mean_std_dev.shape[0], axis=0)
        return np.array(mean, dtype=np.float32), np.array(std_dev, dtype=np.float32)

    @staticmethod
    def load_mean_std_dev_from_stats(file_path, datatype=np.float64):
        stats, sum_length = MeanStdDevExtractor.load_stats(file_path, datatype)

        sum_frames, sum_squared_frames = np.split(stats, stats.shape[0], axis=0)

        mean = sum_frames / sum_length
        std_dev = np.sqrt(sum_squared_frames / sum_length - mean ** 2)

        return np.array(mean, dtype=np.float32), np.array(std_dev, dtype=np.float32)

    @staticmethod
    def combine_stats(file_list, dir_out=None, datatype=np.float64, save_txt=False):
        """
        Combines the stats of different subsets.

        :param file_list:     List where each entry contains the filename (with path) of the files to merge.
        :param dir_out:       If given, saves the combined parameters to that directory as .bin.
        :param datatype:      Format in which the numpy arrays are stored in the files.
        :param save_txt:      Also saves the parameters in a .txt file.
        :return:              Returns the combined normalisation parameters.
        """

        sum_length = 0
        sum_frames = None
        sum_squared_frames = None

        for file in file_list:
            current_stats, labels_len = MeanStdDevExtractor.load_stats(file, datatype=datatype)

            current_sum_frames, current_sum_squared_frames = np.split(current_stats, current_stats.shape[0], axis=0)

            sum_length += labels_len
            if sum_frames is None:
                sum_frames = current_sum_frames
                sum_squared_frames = current_sum_squared_frames
            else:
                sum_frames += current_sum_frames
                sum_squared_frames += current_sum_squared_frames

        stats = np.concatenate((sum_frames, sum_squared_frames), axis=0)

        if dir_out is not None:
            with open(os.path.join(dir_out, MeanStdDevExtractor.file_name_stats + ".bin"), 'wb') as file:
                file.write(struct.pack("i", sum_length))
                np.array(stats, dtype=datatype).tofile(file)

            if save_txt:
                with open(os.path.join(dir_out, MeanStdDevExtractor.file_name_stats + ".txt"), 'w') as file:
                    file.write(str(sum_length) + '\n')
                    np.savetxt(file, stats)

        return sum_length, stats

    @staticmethod
    def combine_mean_std(file_list, dir_out=None, datatype=np.float64, save_txt=True):
        """
        Combines the stats of different subsets.

        :param file_list:     List where each entry contains the filename (with path) of the files to merge.
        :param dir_out:       If given, saves the combined parameters to that directory as .bin.
        :param datatype:      Format in which the numpy arrays are stored in the files.
        :param save_txt:      Also saves the parameters in a .txt file.
        :return:              Returns the combined normalisation parameters.
        """

        length, stats = MeanStdDevExtractor.combine_stats(file_list, dir_out=dir_out, datatype=datatype)

        sum_frames, sum_squared_frames = np.split(stats, stats.shape[0], axis=0)

        mean = sum_frames / length
        std_dev = np.sqrt(sum_squared_frames / length - mean ** 2)

        if dir_out is not None:
            with open(os.path.join(dir_out, MeanStdDevExtractor.file_name_appendix + ".bin"), 'wb') as file:
                file.write(struct.pack("i", length))
                np.concatenate((mean, std_dev), axis=0).tofile(file)

            if save_txt:
                with open(os.path.join(dir_out, MeanStdDevExtractor.file_name_appendix + ".txt"), 'w') as file:
                    file.write(str(length) + '\n')
                    np.savetxt(file, np.concatenate((mean, std_dev), axis=0))

        return mean, std_dev


def main():
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-f", "--file_list", help="List of files containing the parameters.", nargs='+',
                        dest="file_list", required=True)
    parser.add_argument("-o", "--dir_out", help="Directory to save the combined parameters.", type=str,
                        dest="dir_out", required=False, default=None)
    parser.add_argument("-t", "--dtype", help="Format in which numpy array is saved.", default="float64",
                        dest="dtype", choices=["float32", "float64"], required=False)

    # Parse arguments
    args = parser.parse_args()

    # Read the data type.
    if args.dtype == "float32":
        datatype = np.float32
    else:
        datatype = np.float64

    # Combine parameters.
    MeanStdDevExtractor.combine_mean_std(args.file_list, args.dir_out, datatype)


if __name__ == "__main__":
    main()
