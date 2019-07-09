#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#

"""Module description:
   Create atom labels with distribution around amplitudes, a position flag, and vuv flag for .wav files.
"""

# System imports.
import argparse
import logging
import os
import sys
import numpy as np
from scipy.stats import beta, norm

# Third-party imports.

# Local source tree imports.
from idiaptts.src.data_preparation.wcad.AtomLabelGen import AtomLabelGen
from idiaptts.src.data_preparation.world.LF0LabelGen import LF0LabelGen
from idiaptts.misc.normalisation.MeanStdDevExtractor import MeanStdDevExtractor
from idiaptts.misc.utils import surround_with_norm_dist


class AtomVUVDistPosLabelGen(AtomLabelGen):
    """
    Create wcad atom labels with position flag, surrounding distribution around amplitudes, and vuv flag for .wav files.
    """
    logger = logging.getLogger(__name__)

    def __init__(self, wcad_root, dir_atom_labels, dir_world_labels, thetas, k=6,
                 frame_size_ms=5, a_b=3, window_size=51):
        """
        Constructor to use the class as a dataset and data generator.

        :param wcad_root:        Main directory of wcad, containing wcad.py. This directory is added to sys.path.
        :param dir_atom_labels:  While using it as a database dir_atom_labels has to contain the prepared atom labels.
        :param dir_world_labels: While using it as a database dir_world_labels has to contain a folder vuv/ with the
                                 prepared vuv labels.
        :param thetas:           List of theta values.
        :param frame_size_ms:       Length of each frame in ms. Num_frames=total_audio_duration/frame_size_ms, both in ms.
        :param a_b:              A and B value for beta distribution. UNUSED right now.
        :param window_size:      Window of the distribution around each spike in time, should be odd.
        """
        super(AtomVUVDistPosLabelGen, self).__init__(wcad_root, dir_atom_labels, thetas, k, frame_size_ms=frame_size_ms)

        self.dir_world_labels = dir_world_labels
        self.window_size = window_size

        # # Compute distribution coefficients once and later reuse them per frame (only for distribution along theta dim).
        # num_thetas = len(self.theta_interval)
        # self.dist_coefs = np.zeros((num_thetas, num_thetas), dtype=np.float32)
        # for theta_idx in range(num_thetas):
        #     dummy = np.zeros((num_thetas, 1))
        #     dummy[theta_idx] = 1.0
        #     self.dist_coefs[theta_idx] = surround_with_norm_dist(dummy, window_size=3).squeeze()
        #     # self.dist_coefs[theta_idx] = dummy.squeeze()  # No distribution for testing.

    def __getitem__(self, id_name):
        """
        Return the preprocessed sample with the given id_name.
        Only atoms are preprocessed by the preprocess_sample method.
        Labels are trimmed to the shorter one of both (vuv & atoms).
        """
        labels = self.load_sample(id_name, self.dir_labels, len(self.theta_interval), self.dir_world_labels)
        vuv = labels[:, 0, 0:1]
        atoms = labels[:, 1:]
        atoms = self.preprocess_sample(atoms)

        return np.concatenate((vuv, atoms), axis=1)

    def preprocess_sample(self, sample, norm_params=None):
        """
        Normalise one sample (by default to variance 1) and apply a distribution. Add a position flag which has ones
        at the exact atom positions. This function should be used within the batch loading of PyTorch.

        :param sample:            The sample to pre-process.
        :param norm_params:       Use this normalisation parameters instead of self.norm_params.
                                  Default distribution of norm_params is 0 amps, 1 thetas.
        :return:                  Pre-processed sample.
        """

        # Call base class's preprocessing, which does the normalisation.
        sample = super().preprocess_sample(sample, norm_params)

        # Add a position flag.
        pos_flag = np.zeros((len(sample), 1), dtype=np.float32)
        mask = np.any(np.greater(sample, 0), axis=1)
        pos_flag[mask] = 1.0
        mask = np.any(np.less(sample, 0), axis=1)
        pos_flag[mask] = -1.0

        # # Apply distribution in theta space.
        # for idx in range(len(sample)):
        #     sample[idx, :] = (sample[idx, :] * self.dist_coefs).sum(axis=1)
        # Apply distribution in time.
        for theta_idx in range(sample.shape[1]):
            sample[:, theta_idx] = surround_with_norm_dist(sample[:, theta_idx].reshape(-1, 1), self.window_size).squeeze()

        # Fortran order means column-major memory order. This prevent reallocation of memory when the loss function
        # splits sample and position flag.
        return np.asfortranarray(np.concatenate((sample, pos_flag), axis=1))

    def postprocess_sample(self, sample, norm_params=None):
        """
        Identify the peaks in the position flag (remove peaks with absolute value lower than 0.1).
        Set all amplitude outputs to zero except the highest amplitude for positive peaks and the lowest amplitude for
        negative peaks. Then denormalise the amplitudes with the base class method. Set all vuv values < 0.5 to 0 and
        the rest to 1.
        This function is used after inference of a network.

        :param sample:            The sample to post-process.
        :param norm_params:       Use this normalisation parameters instead of self.norm_params.
        :return:                  Post-processed sample.
        """

        # Remove and keep vuv and pos information so that superclass postprocessing can be used.
        vuv = sample[:, 0]
        vuv[vuv < 0.5] = 0.0
        vuv[vuv >= 0.5] = 1.0
        pos = sample[:, -1]
        amps = np.copy(sample[:, 1:-1])

        # Extract atom positions.
        pos = AtomLabelGen.identify_peaks(np.expand_dims(pos, -1), 50)
        pos[abs(pos) < 0.1] = 0

        # Use sign of pos flag for selecting one amplitude.
        amps_max = np.max(amps, axis=1)
        amps_min = np.min(amps, axis=1)
        pos_flag_negative = (pos < 0).squeeze()
        amps_max[pos_flag_negative] = amps_min[pos_flag_negative]  # Use minimum for all negative pos_flags.
        amps_max[(pos == 0).squeeze()] = 0.0
        mask = np.not_equal(np.expand_dims(amps_max, axis=-1).repeat(amps.shape[1], axis=1), amps)
        amps[mask] = 0.0

        # Normalise amplitudes.
        amps = super().postprocess_sample(amps, norm_params)

        # Combine vuv with amps again.
        vuv = np.repeat(vuv[:, np.newaxis, np.newaxis], 2, 2)
        vuv[:, :, 0] = -1  # Set invalid value for lf0.
        return np.concatenate((vuv, amps), axis=1)

    def get_normalisation_params(self, dir_out, file_name=None):
        """
        Read mean std_dev values from a file.
        Save them in self.norm_params

        :param dir_out:       Directory containing the normalisation file.
        :param file_name:     Prefix of normalisation file.
                              Expects file to be named <file_name-><MeanStdDevExtractor.file_name_appendix>.bin
        :return:              Tuple of normalisation parameters (mean, std_dev).
        """

        full_file_name = (file_name + "-" if file_name is not None else "") + MeanStdDevExtractor.file_name_appendix + ".bin"
        self.norm_params = MeanStdDevExtractor.load(os.path.join(self.dir_labels, full_file_name))

        return self.norm_params

    @staticmethod
    def load_sample(id_name, dir_atoms, num_thetas, dir_world):
        """
        Load atoms from dir_atoms/id_name.atom and VUV from dir_world/vuv/id_name.vuv and returns them as tuple.
        """
        id_name = os.path.splitext(os.path.basename(id_name))[0]

        atoms = AtomLabelGen.load_sample(id_name, dir_atoms, num_thetas)
        vuv = LF0LabelGen.load_vuv(id_name, dir_world)

        # Concatenate; do trimming, if necessary.
        if len(atoms) < len(vuv):
            vuv = vuv[:len(atoms)]
        elif len(vuv) < len(atoms):
            atoms = AtomVUVDistPosLabelGen.trim_end_sample(atoms, len(atoms) - len(vuv))

        return np.concatenate((np.repeat(vuv[:, None, :], 2, axis=2), atoms), axis=1)

    @staticmethod
    def labels_to_atoms(np_labels, k=6, frame_size=5, amp_threshold=0.3):
        """
        Transform labels to GammaAtoms.
        Reuse super class method but skip over vuv information in labels.
        """
        return AtomLabelGen.labels_to_atoms(np_labels[:, 1:, :], k, frame_size, amp_threshold)

    def gen_data(self, dir_in, dir_out=None, file_id_list=None, id_list=None, return_dict=False):
        """
        Combines the dictionaries generated by AtomLabelGen and LF0LabelGen.
        The format is T x (1 + |theta|) x 2. The first entry of the second dimension
        is (lf0, vuv) the rest is (amp, theta).

        :return:  Filled dictionary.
        """
        lf0_gen = LF0LabelGen(dir_in, add_deltas=False)

        if return_dict:
            lf0_vuv_dict, *_ = lf0_gen.gen_data(dir_in,
                                                self.dir_world_labels,
                                                file_id_list, id_list,
                                                add_deltas=False, return_dict=True)
            atom_dict, *_ = super().gen_data(dir_in,
                                             self.dir_labels,
                                             file_id_list,
                                             id_list,
                                             return_dict=True)
            # Combine both dicts.
            for id_name, labels in atom_dict.items():
                lf0_vuv = np.expand_dims(lf0_vuv_dict[id_name], axis=1)
                atom_dict[id_name] = np.concatenate((lf0_vuv[:len(labels), :, :], labels), axis=1)

            return atom_dict
        else:
            lf0_gen.gen_data(dir_in,
                             self.dir_world_labels,
                             file_id_list, id_list,
                             add_deltas=False, return_dict=False)
            super().gen_data(dir_in,
                             self.dir_labels,
                             file_id_list,
                             id_list,
                             return_dict=False)


def main():
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-w", "--wcad_root", help="Directory with the WCAD scripts.",
                        type=str, dest="wcad_root", required=True)
    parser.add_argument("-a", "--audio_dir", help="Directory containing the audio (wav) files.",
                        type=str, dest="audio_dir", required=True)
    parser.add_argument("-s", "--sampling_frequency", help="Sampling frequency of all audio files [Hz].",
                        type=int, dest="sampling_frequency", choices=[16000, 48000])
    parser.add_argument("-f", "--frame_size_ms", help="Frame size [ms] of the labels.",
                        type=int, dest="frame_size_ms", default=5)
    parser.add_argument("-i", "--file_id_list_path",
                        help="Path to a text file to read the ids of the files to process. "
                             "Default uses all .wav files in the given audio_dir.",
                        type=str, dest="file_id_list_path", default=None)
    parser.add_argument("-o", "--dir_out", help="Output directory to store the labels.",
                        type=str, dest="dir_out", required=True)
    parser.add_argument("--theta_start", help="Start value of theta.",
                        type=float, dest="theta_start", default=0.01)
    parser.add_argument("--theta_stop", help="Stop value of theta (excluded).",
                        type=float, dest="theta_stop", default=0.055)
    parser.add_argument("--theta_step", help="Distance between the thetas.",
                        type=float, dest="theta_step", default=0.005)

    # Parse arguments
    args = parser.parse_args()

    wcad_root = os.path.abspath(args.wcad_root)
    audio_dir = os.path.abspath(args.audio_dir)
    sampling_frequency = args.sampling_frequency
    frame_size = args.frame_size
    dir_out = os.path.abspath(args.dir_out)

    if frame_size != parser.get_default("frame_size_ms"):
        logging.warning("Frame size must match WCAD configuration in wcad/object_types/params.py.")

    file_id_list = None
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

    theta_start = args.theta_start
    theta_stop = args.theta_stop
    theta_step = args.theta_step

    atom_gen = AtomVUVDistPosLabelGen(wcad_root,
                                      os.path.join(dir_out, "wcad-" + str(theta_start) + "_" + str(theta_stop) + "_" + str(theta_step)),
                                      os.path.join(dir_out, "WORLD"),
                                      np.arange(theta_start, theta_stop, theta_step),
                                      frame_size)
    atom_gen.gen_data(audio_dir, None, file_id_list_name, id_list, return_dict=False)

    # # DEBUG
    # label_dict = atom_gen.gen_data(audio_dir, None, file_id_list_name, id_list, return_dict=True)
    # atom_gen.get_normalisation_params(dir_out, file_id_list_name)
    # test_label = label_dict["roger_5535"]
    # np.set_printoptions(threshold=np.inf)
    # print(test_label[127:130])
    # # The __getitem__ method uses preprocessing only on the atom labels.
    # test_label = np.concatenate((test_label[:, 0, 1:2], atom_gen.preprocess_sample(test_label[:, 1:, ...])), axis=1)
    # print(test_label[127:130])
    # test_label = atom_gen.postprocess_sample(test_label)
    # print(test_label[127:130])

    sys.exit(0)


if __name__ == "__main__":
    main()
