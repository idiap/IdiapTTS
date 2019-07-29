#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#

"""Module description:
   Generate phoneme durations from state-aligned HTK labels.
"""

# System imports.
import argparse
import glob
import logging
import os
import sys
from collections import OrderedDict
import numpy as np

# Third-party imports.

# Local source tree imports.
from idiaptts.misc.normalisation.MeanStdDevExtractor import MeanStdDevExtractor
from idiaptts.misc.normalisation.MinMaxExtractor import MinMaxExtractor
from idiaptts.src.data_preparation.LabelGen import LabelGen
from idiaptts.misc.utils import makedirs_safe, interpolate_lin, compute_deltas


class PhonemeDurationLabelGen(LabelGen):
    """Load phoneme durations from state-aligned HTK labels."""

    ext_phonemes = ".lab"  # Extension of HTK labels files.
    ext_durations = ".dur"  # Extension of created duration labels.
    num_states = 5  # Number of states per phoneme in HTK labels.
    min_phoneme_length = 50000  # Divisor for phoneme timings in HTK labels.
    logger = logging.getLogger(__name__)

    def __init__(self, dir_labels):

        self.dir_labels = dir_labels
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
        batch loading.

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
    def load_sample(id_name, dir_out):
        """
        Load durations from dir_out.

        :param id_name:         Id of the sample.
        :param dir_out:         Directory containing the sample.
        :return:                Numpy array with dimensions num_frames x self.num_states.
        """
        logging.debug("Load duration for " + id_name)

        with open(os.path.join(dir_out, id_name + PhonemeDurationLabelGen.ext_durations), 'r') as f:
            dur = np.fromfile(f, dtype=np.float32)
            dur = np.reshape(dur, [-1, PhonemeDurationLabelGen.num_states])

        return dur

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
        self.norm_params = MeanStdDevExtractor.load(os.path.join(dir_out, full_file_name))

        return self.norm_params

    def gen_data(self, dir_in, dir_out=None, file_id_list=None, id_list=None, return_dict=False):
        """
        Prepare durations from HTK labels (forced-aligned).
        Each numpy array has the dimension num_phonemes x PhonemeDurationLabelGen.num_states (default num_state=5).

        :param dir_in:         Directory where the HTK label files are stored (usually named label_state_align).
        :param dir_out:        Main directory where the labels and normalisation parameters are saved to.
                               If None, labels are not saved.
        :param file_id_list:   Name of the file containing the ids. Normalisation parameters are saved using
                               this name to differentiate parameters between subsets.
        :param id_list:        The list of utterances to process.
                               Should have the form uttId1 \\n uttId2 \\n ...\\n uttIdN.
                               If None, all file in dir_in are used.
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
            makedirs_safe(dir_out)

        # Create the return dictionary if required.
        if return_dict:
            label_dict = OrderedDict()

        # Create normalisation computation units.
        norm_params_ext_dur = MeanStdDevExtractor()

        logging.info("Extract phoneme durations for " + "[{0}]".format(", ".join(str(i) for i in id_list)))
        for file_name in id_list:
            logging.debug("Extract phoneme durations from " + file_name)

            with open(os.path.join(dir_in, file_name + PhonemeDurationLabelGen.ext_phonemes), 'r') as f:
                htk_labels = [line.rstrip('\n').split()[:2] for line in f]
                timings = np.array(htk_labels, dtype=np.float32) / PhonemeDurationLabelGen.min_phoneme_length
                dur = timings[:, 1] - timings[:, 0]
                dur = dur.reshape(-1, PhonemeDurationLabelGen.num_states).astype(np.float32)

            if return_dict:
                label_dict[file_name] = dur
            if dir_out is not None:
                dur.tofile(os.path.join(dir_out, file_name + PhonemeDurationLabelGen.ext_durations))

            # Add sample to normalisation computation unit.
            norm_params_ext_dur.add_sample(dur)

        # Save mean and std dev of all features.
        norm_params_ext_dur.save(os.path.join(dir_out, file_id_list_name))

        # Get normalisation parameters.
        norm_first, norm_second = norm_params_ext_dur.get_params()

        if return_dict:
            # Return dict of labels for all utterances.
            return label_dict, norm_first, norm_second
        else:
            return norm_first, norm_second


def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-l", "--dir_labels", help="Directory containing the label (HTK full labels, *.lab) files.",
                        type=str, dest="dir_labels", required=True)
    parser.add_argument("-i", "--file_id_list_path",
                        help="Path to text file to read the ids of the files to process.\
                              Default uses all .wav files in the given audio_dir.",
                        type=str, dest="file_id_list_path", default=None)
    parser.add_argument("-o", "--dir_out", help="Output directory to store the labels."
                                                "Within the output directory folders for each feature will be created.",
                        type=str, dest="dir_out", required=True)

    # Parse arguments
    args = parser.parse_args()

    dir_out = os.path.abspath(args.dir_out)
    dir_labels = os.path.abspath(args.dir_labels)

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
    dur_gen = PhonemeDurationLabelGen(dir_out)
    dur_gen.gen_data(dir_labels, dir_out, args.file_id_list_path, id_list, return_dict=False)

    # # DEBUG
    # label_dict, *_ = dur_gen.gen_data(dir_labels, dir_out, args.file_id_list_path, id_list, return_dict=True)
    # # Call this once before starting the pre-processing.
    # norm_params = dur_gen.get_normalisation_params(dir_out, file_name=file_id_list_name)
    #
    # test_label = label_dict["roger_5535"]
    # print(test_label)
    # test_label_pre = dur_gen.preprocess_sample(test_label)
    # assert((test_label_pre == dur_gen["roger_5535"]).all())
    # print(test_label)
    # test_label_post = dur_gen.postprocess_sample(test_label_pre)
    # print(test_label_post)
    # assert((test_label == test_label_post).all())

    sys.exit(0)


if __name__ == "__main__":
    main()
