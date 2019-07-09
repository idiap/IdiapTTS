#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#

# System imports.
import os
import logging
import argparse
import numpy as np


# Third-party imports.

# Local source tree imports.


class MinMaxExtractor(object):
    """A class to collect minimum and maximum of data points.
    Calling it as a script is used to combine min and max of different subsets.
    """
    logger = logging.getLogger(__name__)

    # Constants.
    file_name_appendix = "min-max"

    def __init__(self):
        self.combined_min = None
        self.combined_max = None

    def add_sample(self, sample):

        if self.combined_min is None:
            self.combined_min = np.copy(sample.min(axis=0))
            self.combined_max = np.copy(sample.max(axis=0))
        else:
            self.combined_min = np.minimum(self.combined_min, sample.min(axis=0))
            self.combined_max = np.maximum(self.combined_max, sample.max(axis=0))

    def get_params(self):
        return self.combined_min, self.combined_max

    def save(self, filename, datatype=np.float64):
        samples_min, samples_max = self.get_params()
        samples_min_max = np.stack((samples_min, samples_max), axis=0)

        if datatype is np.str:
            np.savetxt(filename + "-" + self.file_name_appendix + ".txt", samples_min_max)
        elif datatype is np.float32 or datatype is np.float64:
            with open(filename + "-" + self.file_name_appendix + ".bin", 'wb') as file:
                np.array(samples_min_max, dtype=datatype).tofile(file)
        else:
            self.logger.error("Unknown datatype: " + datatype.__name__ + ". "
                              "Please choose one of {numpy.float32, numpy.float64, numpy.str}.")

    @staticmethod
    def load(file_path, datatype=np.float64):
        if datatype is np.str:
            min_max = np.loadtxt(file_path, dtype=np.float32).reshape((2, -1))
        elif datatype is np.float32 or datatype is np.float64:
            with open(file_path, 'rb') as f:
                min_max = np.fromfile(f, dtype=datatype).reshape((2, -1))
        else:
            logging.error("Unknown datatype: " + datatype.__name__ + ". "
                          "Please choose one of {numpy.float32, numpy.float64, numpy.str}.")
            return None

        return min_max

    @staticmethod
    def combine_min_max(file_list, dir_out=None, datatype=np.float64, save_txt=False):
        """
        Computes the combined min and max from all min and max of the subgroups.

        :param file_list:     List where each entry contains the filename (with path) of the files to merge.
        :param dir_out:       If given, saves the combined parameters to that directory as .bin.
        :param datatype:      Format in which the numpy arrays are stored in the files.
        :param save_txt:      Also saves the parameters in a .txt file.
        :return:              Returns the combined normalisation parameters.
        """
        samples_minimum = None
        samples_maximum = None

        for file in file_list:
            current_min_max = MinMaxExtractor.load(file, datatype=datatype)
            current_min, current_max = np.split(current_min_max, current_min_max.shape[0], axis=0)
            current_min = current_min.squeeze()
            current_max = current_max.squeeze()

            if samples_maximum is None:
                samples_minimum = current_min
                samples_maximum = current_max
            else:
                samples_minimum = np.minimum(samples_minimum, current_min)
                samples_maximum = np.maximum(samples_maximum, current_max)

        samples_min_max = np.stack((samples_minimum, samples_maximum), axis=0)

        if dir_out is not None:
            with open(os.path.join(dir_out, MinMaxExtractor.file_name_appendix + ".bin"), 'wb') as file:
                np.array(samples_min_max, dtype=datatype).tofile(file)

            if save_txt:
                np.savetxt(os.path.join(dir_out, MinMaxExtractor.file_name_appendix + ".txt"), samples_min_max)

        return samples_min_max



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
    else:  # args.dtype == "float64":
        datatype = np.float64

    # Combine parameters.
    MinMaxExtractor.combine_min_max(args.file_list, args.dir_out, datatype)


if __name__ == "__main__":
    main()
