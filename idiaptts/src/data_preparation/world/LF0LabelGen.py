#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#

"""Module description:
   Create world lf0 and vuv feature labels for .wav files.
"""

# System imports.
import argparse
import glob
import logging
import math
import os
import sys
from collections import OrderedDict
import numpy as np

# Third-party imports.
import pyworld
import soundfile

# Local source tree imports.
from idiaptts.misc.normalisation.MeanStdDevExtractor import MeanStdDevExtractor
from idiaptts.misc.normalisation.MinMaxExtractor import MinMaxExtractor
from idiaptts.src.data_preparation.LabelGen import LabelGen
from idiaptts.misc.utils import makedirs_safe, interpolate_lin, compute_deltas


class LF0LabelGen(LabelGen):
    """Create LF0 feature labels for .wav files."""
    f0_silence_threshold = 20
    lf0_zero = 0

    dir_lf0 = "lf0"
    dir_deltas = "lf0"
    dir_vuv = "vuv"
    ext_lf0 = ".lf0"
    ext_deltas = ".lf0_deltas"
    ext_vuv = ".vuv"

    logger = logging.getLogger(__name__)

    def __init__(self, dir_labels, add_deltas=False):
        """
        Prepare a numpy array with the LF0 and V/UV labels for each frame for each utterance extracted by WORLD.
        If add_delta is false each frame has only the LF0 value, otherwise its deltas and double deltas are added.

        :param dir_labels:             While using it as a database dir_labels has to contain the prepared labels.
        :param add_deltas:             Determines if labels contain deltas and double deltas.
        """
        # Attributes.
        self.dir_labels = dir_labels
        self.add_deltas = add_deltas
        self.norm_params = None

    def __getitem__(self, id_name):
        """Return the preprocessed sample with the given id_name."""
        sample = self.load_sample(id_name, self.dir_labels)
        sample = self.preprocess_sample(sample)

        return sample

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
        """
        Normalise one sample (by default to 0 mean and variance 1). This function should be used within the
        batch loading of PyTorch.

        :param sample:            The sample to pre-process.
        :param norm_params:       Use this normalisation parameters instead of self.norm_params.
        :return:                  Pre-processed sample.
        """

        if norm_params is not None:
            mean, std_dev = norm_params
        elif self.norm_params is not None:
            mean, std_dev = self.norm_params
        else:
            self.logger.error("Please give norm_params argument or call get_normaliations_params() before.")
            return None

        return np.float32((sample - mean) / std_dev)

    def postprocess_sample(self, sample, norm_params=None):
        """
        Denormalise one sample. This function is used after inference of a network.

        :param sample:            The sample to post-process.
        :param norm_params:       Use this normalisation parameters instead of self.norm_params.
        :return:                  Post-processed sample.
        """
        if norm_params is not None:
            mean, std_dev = norm_params
        elif self.norm_params is not None:
            mean, std_dev = self.norm_params
        else:
            self.logger.error("Please give norm_params argument or call get_normaliations_params() before.")
            return None

        sample = np.copy((sample * std_dev) + mean)

        return sample

    @staticmethod
    def load_sample(id_name, dir_out, add_deltas=False):
        """
        Load LF0 and V/UV features from dir_out.

        :param id_name:         Id of the sample.
        :param dir_out:         Directory containing the sample.
        :param add_deltas:      Determines if deltas and double deltas are expected.
        :return:                Numpy array with dimensions num_frames x len(lf0, vuv).
        """
        logging.debug("Load WORLD features for " + id_name)

        lf0 = LF0LabelGen.load_lf0(id_name, dir_out, add_deltas)
        vuv = LF0LabelGen.load_vuv(id_name, dir_out)
        labels = np.concatenate((lf0, vuv), axis=1)

        return labels

    @staticmethod
    def load_lf0(id_name, dir_out, add_deltas=False):
        """Loads LF0 features from dir_out."""
        if add_deltas:
            with open(os.path.join(dir_out, LF0LabelGen.dir_lf0, id_name + LF0LabelGen.ext_deltas), 'rb') as f:
                lf0 = np.fromfile(f, dtype=np.float32)
                lf0 = np.reshape(lf0, [-1, 3])
        else:
            with open(os.path.join(dir_out, LF0LabelGen.dir_lf0, id_name + LF0LabelGen.ext_lf0), 'rb') as f:
                lf0 = np.fromfile(f, dtype=np.float32)
                lf0 = np.reshape(lf0, [-1, 1])
        return lf0

    @staticmethod
    def load_vuv(id_name, dir_out):
        """Loads V/UV features from dir_out."""
        dim_vuv = 1
        with open(os.path.join(dir_out, LF0LabelGen.dir_vuv, id_name + LF0LabelGen.ext_vuv), 'rb') as f:
            vuv = np.fromfile(f, dtype=np.float32)
            vuv = np.reshape(vuv, [-1, dim_vuv])
        return vuv

    @staticmethod
    def convert_to_world_features(sample):
        lf0 = sample[:, 0]
        vuv = np.copy(sample[:, -1])
        vuv[vuv < 0.5] = 0.0
        vuv[vuv >= 0.5] = 1.0

        return lf0, vuv

    def get_normalisation_params(self, dir_out, file_name=None):
        """
        Read the mean std_dev values from a file.
        Save them in self.norm_params.

        :param dir_out:       Directory containing the normalisation file.
        :param file_name:     Prefix of normalisation file.
                              Expects file to be named <file_name-><MeanStdDevExtractor.file_name_appendix>.bin
        :return:              Tuple of normalisation parameters (mean, std_dev).
        """

        full_file_name = (file_name + "-" if file_name is not None else "") + MeanStdDevExtractor.file_name_appendix + ".bin"

        if not self.add_deltas:
            # Collect all means and std_devs in a list.
            all_mean = list()
            all_std_dev = list()
            # Load normalisation parameters for all features.
            mean, std_dev = MeanStdDevExtractor.load(os.path.join(dir_out, self.dir_lf0, full_file_name))
            all_mean.append(np.atleast_2d(mean))
            all_std_dev.append(np.atleast_2d(std_dev))
            # Manually set vuv normalisation parameters.
            # Note that vuv normalisation parameters are not saved in gen_data method (except for add_deltas=True).
            all_mean.append(np.atleast_2d(0.0))
            all_std_dev.append(np.atleast_2d(1.0))

            # for dir_feature in [self.dir_lf0, self.dir_vuv]:
            #     mean, std_dev = MeanStdDevExtractor.load(os.path.join(dir_out, dir_feature, full_file_name))
            #     all_mean.append(np.atleast_2d(mean))
            #     all_std_dev.append(np.atleast_2d(std_dev))

            # Save the concatenated normalisation parameters locally.
            self.norm_params = np.concatenate(all_mean, axis=1), np.concatenate(all_std_dev, axis=1)
        else:
            # Save the normalisation parameters locally.
            # VUV normalisation parameters are manually set to mean=0 and std_dev=1 in gen_data method.
            self.norm_params = MeanStdDevExtractor.load(os.path.join(dir_out, self.dir_deltas, full_file_name))

        return self.norm_params

    def gen_data(self, dir_in, dir_out=None, file_id_list=None, id_list=None, add_deltas=False, return_dict=False):
        """
        Prepare LF0 and V/UV features from audio files. If add_delta is false each numpy array has the dimension
        num_frames x 2 [f0, vuv], otherwise the deltas and double deltas are added between
        the features resulting in num_frames x 4 [lf0(3*1), vuv].

        :param dir_in:         Directory where the .wav files are stored for each utterance to process.
        :param dir_out:        Main directory where the labels and normalisation parameters are saved to subdirectories.
                               If None, labels are not saved.
        :param file_id_list:   Name of the file containing the ids. Normalisation parameters are saved using
                               this name to differentiate parameters between subsets.
        :param id_list:        The list of utterances to process.
                               Should have the form uttId1 \\n uttId2 \\n ...\\n uttIdN.
                               If None, all file in audio_dir are used.
        :param add_deltas:     Add deltas and double deltas to all features except vuv.
        :param return_dict:    If true, returns an OrderedDict of all samples as first output.
        :return:               Returns two normalisation parameters as tuple. If return_dict is True it returns
                               all processed labels in an OrderedDict followed by the two normalisation parameters.
        """

        # Fill file_id_list by .wav files in dir_in if not given and set an appropriate file_id_list_name.
        if id_list is None:
            id_list = list()
            filenames = glob.glob(os.path.join(dir_in, "*.wav"))
            for filename in filenames:
                id_list.append(os.path.splitext(os.path.basename(filename))[0])
            file_id_list_name = "all"
        else:
            file_id_list_name = os.path.splitext(os.path.basename(file_id_list))[0]

        # Create directories in dir_out if it is given.
        if dir_out is not None:
            if add_deltas:
                makedirs_safe(os.path.join(dir_out, LF0LabelGen.dir_deltas))
            else:
                makedirs_safe(os.path.join(dir_out, LF0LabelGen.dir_lf0))
                makedirs_safe(os.path.join(dir_out, LF0LabelGen.dir_vuv))

        # Create the return dictionary if required.
        if return_dict:
            label_dict = OrderedDict()

        # Create normalisation computation units.
        norm_params_ext_lf0 = MeanStdDevExtractor()
        # norm_params_ext_vuv = MeanStdDevExtractor()
        norm_params_ext_deltas = MeanStdDevExtractor()

        logging.info("Extract WORLD LF0 features for " + "[{0}]".format(", ".join(str(i) for i in id_list)))
        for file_name in id_list:
            logging.debug("Extract WORLD LF0 features from " + file_name)

            # Load audio file and extract features.
            audio_name = os.path.join(dir_in, file_name + ".wav")
            raw, fs = soundfile.read(audio_name)
            _f0, t = pyworld.dio(raw, fs)  # Raw pitch extraction. TODO: Use magphase here?
            f0 = pyworld.stonemask(raw, _f0, t, fs)  # Pitch refinement.

            # Compute lf0 and vuv information.
            lf0 = np.log(f0, dtype=np.float32)
            lf0[lf0 <= math.log(LF0LabelGen.f0_silence_threshold)] = LF0LabelGen.lf0_zero
            lf0, vuv = interpolate_lin(lf0)

            if add_deltas:
                # Compute the deltas and double deltas for all features.
                lf0_deltas, lf0_double_deltas = compute_deltas(lf0)

                # Combine them to a single feature sample.
                labels = np.concatenate((lf0, lf0_deltas, lf0_double_deltas, vuv), axis=1)

                # Save into return dictionary and/or file.
                if return_dict:
                    label_dict[file_name] = labels
                if dir_out is not None:
                    labels.tofile(os.path.join(dir_out, LF0LabelGen.dir_deltas, file_name + LF0LabelGen.ext_deltas))

                # Add sample to normalisation computation unit.
                norm_params_ext_deltas.add_sample(labels)
            else:
                # Save into return dictionary and/or file.
                if return_dict:
                    label_dict[file_name] = np.concatenate((lf0, vuv), axis=1)
                if dir_out is not None:
                    lf0.tofile(os.path.join(dir_out, LF0LabelGen.dir_lf0, file_name + LF0LabelGen.ext_lf0))
                    vuv.astype(np.float32).tofile(os.path.join(dir_out, LF0LabelGen.dir_vuv, file_name + LF0LabelGen.ext_vuv))

                # Add sample to normalisation computation unit.
                norm_params_ext_lf0.add_sample(lf0)
                # norm_params_ext_vuv.add_sample(vuv)

        # Save mean and std dev of all features.
        if not add_deltas:
            norm_params_ext_lf0.save(os.path.join(dir_out, LF0LabelGen.dir_lf0, file_id_list_name))
            # norm_params_ext_vuv.save(os.path.join(dir_out, LF0LabelGen.dir_vuv, file_id_list_name))
        else:
            # Manually set vuv normalisation parameters before saving.
            norm_params_ext_deltas.sum_frames[-1] = 0.0  # Mean = 0.0
            norm_params_ext_deltas.sum_squared_frames[-1] = norm_params_ext_deltas.sum_length  # Variance = 1.0
            norm_params_ext_deltas.save(os.path.join(dir_out, LF0LabelGen.dir_deltas, file_id_list_name))

        # Get normalisation parameters.
        if not add_deltas:
            norm_lf0 = norm_params_ext_lf0.get_params()
            # norm_vuv = norm_params_ext_vuv.get_params()

            norm_first = np.concatenate((norm_lf0[0], (0.0,)), axis=0)
            norm_second = np.concatenate((norm_lf0[1], (1.0,)), axis=0)
        else:
            norm_first, norm_second = norm_params_ext_deltas.get_params()

        if return_dict:
            # Return dict of labels for all utterances.
            return label_dict, norm_first, norm_second
        else:
            return norm_first, norm_second


def main():
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-a", "--dir_audio", help="Directory containing the audio (wav) files.",
                        type=str, dest="dir_audio", required=True)
    # parser.add_argument("-s", "--sampling_frequency", help="Sampling frequency of all audio files [Hz].",
    #                     type=int, dest="sampling_frequency", choices=[16000, 48000])
    parser.add_argument("-i", "--file_id_list_path",
                        help="Path to text file to read the ids of the files to process.\
                              Default uses all .wav files in the given audio_dir.",
                        type=str, dest="file_id_list_path", default=None)
    parser.add_argument("-o", "--dir_out", help="Output directory to store the labels."
                                                "Within the output directory folders for each feature will be created.",
                        type=str, dest="dir_out", required=True)
    parser.add_argument("--add_deltas", help="Defines if features are augmented by their deltas and double deltas."
                                             "Features will then be stored as a single file.",
                        dest="add_deltas", action='store_const', const=True, default=False)

    # Parse arguments
    args = parser.parse_args()

    dir_audio = os.path.abspath(args.dir_audio)
    dir_out = os.path.abspath(args.dir_out)

    # Read ids and select an appropriate file_id_list_name,
    # used to identify normalisation parameters of different subsets.
    if args.file_id_list_path is not None:
        file_id_list_path = os.path.abspath(args.file_id_list_path)
        # Read which files to process.
        with open(file_id_list_path) as f:
            id_list = f.readlines()
        # Trim entries in-place.
        id_list[:] = [s.strip(' \t\n\r') for s in id_list]
        file_id_list_name = os.path.splitext(os.path.basename(file_id_list_path))[0]
    else:
        id_list = None
        file_id_list_name = "all"

    # Execute main functionality.
    lf0_gen = LF0LabelGen(dir_out, args.add_deltas)
    lf0_gen.gen_data(dir_audio,
                     dir_out=dir_out,
                     file_id_list=args.file_id_list_path,
                     id_list=id_list,
                     add_deltas=args.add_deltas,
                     return_dict=False)

    # # DEBUG
    # label_dict, norm_first, norm_second = lf0_gen.gen_data(dir_audio,
    #                                                        dir_out=dir_out,
    #                                                        file_id_list=args.file_id_list_path,
    #                                                        id_list=id_list,
    #                                                        add_deltas=args.add_deltas,
    #                                                        return_dict=True)
    #
    # # Call this once before starting the preprocessing.
    # lf0_gen.get_normalisation_params(dir_out, file_name=file_id_list_name)
    # test_label = label_dict["roger_5535"]
    # print(test_label[98:102])
    # test_label = lf0_gen.preprocess_sample(test_label)
    # print(test_label[98:102])
    # test_label = lf0_gen.postprocess_sample(test_label)
    # print(test_label[98:102])

    sys.exit(0)


if __name__ == "__main__":
    main()
