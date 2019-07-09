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


class MeanCovarianceExtractor(object):
    """A class to collect mean and covariance matrix of data in an online manner.
    Calling it as a script is used to combine mean and covariance of different subsets.
    """
    logger = logging.getLogger(__name__)

    # Constants.
    file_name_stats = "stats"
    file_name_appendix = "mean-covariance"

    def __init__(self):
        self.sum_length = 0
        self.sum_frames = 0
        self.sum_product_frames = 0

    def add_sample(self, sample):

        self.sum_length += len(sample)

        self.sum_frames += np.sum(sample, axis=0, keepdims=True)
        self.sum_product_frames += np.dot(np.transpose(sample), sample)

    def get_params(self):
        mean = self.sum_frames / self.sum_length
        mean_product = np.dot(np.transpose(mean), mean)
        covariance = self.sum_product_frames / self.sum_length - mean_product

        return np.atleast_2d(mean, covariance)

    def save(self, filename, datatype=np.float64):
        self.save_stats(filename, datatype)
        self.save_mean_covariance(filename, datatype)

    def save_stats(self, filename, datatype=np.float64):

        stats = np.concatenate((np.atleast_1d(self.sum_frames), np.atleast_1d(self.sum_product_frames)))

        if datatype is np.str:
            np.savetxt(filename + "-" + self.file_name_stats + ".txt", stats, header=str(self.sum_length))
        elif datatype is np.float32 or datatype is np.float64:
            with open(filename + "-" + self.file_name_stats + ".bin", 'wb') as file:
                file.write(struct.pack("ii", self.sum_length, len(stats)))
                np.array(stats, dtype=datatype).tofile(file)
        else:
            self.logger.error("Unknown datatype: " + datatype.__name__ + ". "
                              "Please choose one of {numpy.float32, numpy.float64, numpy.str}.")

    def save_mean_covariance(self, filename, datatype=np.float64):
        mean_covariance = np.concatenate((self.get_params()), axis=0)

        if datatype is np.str:
            np.savetxt(filename + "-" + self.file_name_appendix + ".txt", mean_covariance, header=str(self.sum_length))
        elif datatype is np.float32 or datatype is np.float64:
            with open(filename + "-" + self.file_name_appendix + ".bin", 'wb') as file:
                file.write(struct.pack("ii", self.sum_length, len(mean_covariance)))
                np.array(mean_covariance, dtype=datatype).tofile(file)
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
                header = struct.unpack("ii", f.read(8))
                labels_len = header[0]
                size = header[1]
                stats = np.fromfile(f, dtype=datatype).reshape((size, -1))
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
                mean_covariance = np.loadtxt(f, dtype=np.float32)
        elif datatype is np.float32 or datatype is np.float64:
            with open(file_path, 'rb') as f:
                header = struct.unpack("ii", f.read(8))
                labels_len = header[0]
                size = header[1]
                mean_covariance = np.fromfile(f, dtype=datatype).reshape((size, -1))
        else:
            logging.error("Unknown datatype: " + datatype.__name__ + ". "
                          "Please choose one of {numpy.float32, numpy.float64, numpy.str}.")
            return None

        mean, covariance = np.split(mean_covariance, (1,), axis=0)
        std_dev = np.sqrt(np.diag(covariance), dtype=np.float32)

        return np.array(mean.squeeze(), dtype=np.float32), np.atleast_2d(np.array(covariance, dtype=np.float32)), std_dev.squeeze()

    @staticmethod
    def load_mean_covariance_from_stats(file_path, datatype=np.float64):
        stats, sum_length = MeanCovarianceExtractor.load_stats(file_path, datatype)

        sum_frames, sum_product_frames = np.split(stats, (1,), axis=0)

        mean = sum_frames / sum_length
        mean_product = np.dot(np.transpose(mean), mean)
        covariance = sum_product_frames / sum_length - mean_product

        return np.atleast_2d(np.array(mean, dtype=np.float32), np.array(covariance, dtype=np.float32))

    @staticmethod
    def combine_stats(file_list, dir_out=None, datatype=np.float64, file_name = '', save_txt=False):
        """
        Combines the stats of different subsets.

        :param file_list:     List where each entry contains the filename (with path) of the files to merge.
        :param dir_out:       If given, saves the combined parameters to that directory as .bin.
        :param datatype:      Format in which the numpy arrays are stored in the files.
        :param save_txt:      Also saves the parameters in a .txt file.
        :return:              Returns the combined normalisation parameters.
        """

        sum_length = 0
        sum_frames = 0
        sum_product_frames = 0

        for file in file_list:
            current_stats, labels_len = MeanCovarianceExtractor.load_stats(file, datatype=datatype)

            current_sum_frames, current_sum_product_frames = np.split(current_stats, (1,), axis=0)

            sum_length += labels_len

            sum_frames += current_sum_frames
            sum_product_frames += current_sum_product_frames

        stats = np.concatenate((sum_frames, sum_product_frames), axis=0)

        if dir_out is not None:
            with open(os.path.join(dir_out, MeanCovarianceExtractor.file_name_stats + file_name + ".bin"), 'wb') as file:
                file.write(struct.pack("ii", sum_length, len(stats)))
                np.array(stats, dtype=datatype).tofile(file)

            if save_txt:
                with open(os.path.join(dir_out, MeanCovarianceExtractor.file_name_stats + file_name + ".txt"), 'w') as file:
                    file.write(str(sum_length) + '\n')
                    np.savetxt(file, stats)

        return sum_length, stats

    @staticmethod
    def combine_mean_covariance(file_list, dir_out=None, file_name='', datatype=np.float64, save_txt=True):
        """
        Combines the stats of different subsets.

        :param file_list:     List where each entry contains the filename (with path) of the files to merge.
        :param dir_out:       If given, saves the combined parameters to that directory as .bin.
        :param datatype:      Format in which the numpy arrays are stored in the files.
        :param save_txt:      Also saves the parameters in a .txt file.
        :return:              Returns the combined normalisation parameters.
        """

        length, stats = MeanCovarianceExtractor.combine_stats(file_list, dir_out=dir_out, file_name=file_name, datatype=datatype)

        sum_frames, sum_product_frames = np.split(stats, (1,), axis=0)

        mean = sum_frames / length
        mean_product = np.dot(np.transpose(mean), mean)
        covariance = sum_product_frames / length - mean_product
        combined = np.concatenate((mean, covariance), axis=0)

        if dir_out is not None:
            with open(os.path.join(dir_out, MeanCovarianceExtractor.file_name_appendix + file_name + ".bin"), 'wb') as file:
                file.write(struct.pack("ii", length, len(combined)))
                combined.tofile(file)

            if save_txt:
                with open(os.path.join(dir_out, MeanCovarianceExtractor.file_name_appendix + file_name + ".txt"), 'w') as file:
                    file.write(str(length) + '\n')
                    np.savetxt(file, combined)

        return np.atleast_2d(mean, covariance)


def main():
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-f", "--file_list", help="List of files containing the parameters.", nargs='+',
                        dest="file_list", required=True)
    parser.add_argument("-o", "--dir_out", help="Directory to save the combined parameters.", type=str,
                        dest="dir_out", required=False, default=None)
    parser.add_argument("-n", "--file_name", help="File name to save.", type=str,
                        dest="file_name", required=False, default='')
    parser.add_argument("-t", "--dtype", help="Format in which numpy array is saved.", default="float64",
                        dest="dtype", choices=["float32", "float64"], required=False)

    # Parse arguments
    args = parser.parse_args()

    # Read the data type.
    if args.dtype == "float32":
        datatype = np.float32
    else:  # args.dtype == "float64":
        datatype = np.float64

    # Combine parameters.
    MeanCovarianceExtractor.combine_mean_covariance(args.file_list, args.dir_out, datatype=datatype, file_name=args.file_name)


if __name__ == "__main__":
    main()
