#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#

"""Module description:
   Create world feature labels for .wav files.
"""


# System imports.
import argparse
import glob
import logging
import numpy as np
import os
import math
import sys
from collections import OrderedDict

# Third-party imports.
import pyworld
import soundfile
import pysptk

# Local source tree imports.
from idiaptts.misc.normalisation.MeanStdDevExtractor import MeanStdDevExtractor
from idiaptts.misc.normalisation.MeanCovarianceExtractor import MeanCovarianceExtractor
from idiaptts.src.data_preparation.LabelGen import LabelGen
from idiaptts.misc.utils import makedirs_safe, interpolate_lin, compute_deltas
from idiaptts.misc.mlpg import MLPG


class WorldFeatLabelGen(LabelGen):
    """Create world feat labels for .wav files."""
    # TODO: Constants into hparams.
    f0_silence_threshold = 30
    lf0_zero = 0
    mgc_alpha = 0.58

    dir_lf0 = "lf0"
    dir_vuv = "vuv"
    dir_coded_sps = "mgc"
    # dir_coded_sps = "mfcc"
    dir_bap = "bap"
    dir_deltas = "cmp"

    ext_lf0 = ".lf0"
    ext_vuv = ".vuv"
    ext_coded_sp = ".mgc"
    # ext_coded_sp = ".mfcc"
    ext_bap = ".bap"
    ext_deltas = ".cmp"

    logger = logging.getLogger(__name__)

    def __init__(self, dir_labels, add_deltas=False, sampling_fn=None, num_coded_sps=60):
        """
        Constructor to use the class as a database.
        If add_delta is false labels have the dimension num_frames x (num_coded_sps + 3) [mgc(num_coded_sps), lf0, vuv, bap(1)],
        otherwise the deltas and double deltas are added between the features resulting in num_frames x (3*num_coded_sps + 7)
        [mgc(3*num_coded_sps), lf0(3*1), vuv, bap(3*1)].

        :param dir_labels:             While using it as a database dir_labels has to contain the prepared labels.
        :param add_deltas:             Determines if labels contain deltas and double deltas.
        :param sampling_fn:            Provide a function for up- or down-sampling the features during preprocessing.
        :param num_coded_sps:          Number of bins used to represent the coded spectral features.
        """
        # TODO: Select frequency representation: mfcc, mgc, mel filter banks. Current implementation uses mgc.

        # Attributes.
        self.dir_labels = dir_labels  # Only used in __getitem__().
        self.add_deltas = add_deltas
        self.sampling_fn = sampling_fn
        self.norm_params = None
        self.cov_coded_sp = None
        self.cov_lf0 = None
        self.cov_bap = None
        self.num_coded_sps = num_coded_sps
        self.dir_coded_sps += str(num_coded_sps)
        self.dir_deltas += "_" + self.dir_coded_sps
        # self.dir_deltas += "_{}sp".format(num_coded_sps)  # TODO: Use this instead.

    def __getitem__(self, id_name):
        """Return the preprocessed sample with the given id_name."""
        sample = self.load_sample(id_name, self.dir_labels, self.add_deltas, num_coded_sps=self.num_coded_sps)
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
                                  Default distribution of norm_params for self.add_deltas=False is
                                  mgc, lf0, vuv, bap.
                                  Default distribution of norm_params for self.add_deltas=True is
                                  mgc, mgc deltas, mgc double deltas,
                                  lf0, lf0 deltas, lf0 double deltas,
                                  vuv,
                                  bap, bap deltas, bap double deltas.
        :return:                  Pre-processed sample.
        """

        if norm_params is not None:
            mean, std_dev = norm_params
        elif self.norm_params is not None:
            mean, std_dev = self.norm_params
        else:
            self.logger.error("Please give norm_params argument or call get_normaliations_params() before.")
            raise ValueError("Please give norm_params argument or call get_normalisation_params() before.")

        sample = np.float32((sample - mean) / std_dev)

        if self.sampling_fn is not None:
            sample = self.sampling_fn(sample)

        return sample

    def postprocess_sample(self, sample, norm_params=None, apply_mlpg=True):
        """
        Postprocess one sample. This function is used after inference of a network.

        :param sample:            The sample to post-process.
        :param norm_params:       Use this normalisation parameters instead of self.norm_params.
        :param apply_mlpg:        Apply the MLPG algorithm on the post-processed sample.
        :return:                  Post-processed sample.
        """
        if norm_params is not None:
            mean, std_dev = norm_params
        elif self.norm_params is not None:
            mean, std_dev = self.norm_params
        else:
            self.logger.error("Please give norm_params argument or call get_normalisation_params() before.")
            raise ValueError("Please give norm_params argument or call get_normalisation_params() before.")

        sample = np.copy((sample * std_dev) + mean)

        if self.add_deltas:
            coded_sp_full = sample[:, :self.num_coded_sps * 3]
            lf0_full = sample[:, self.num_coded_sps * 3:self.num_coded_sps * 3 + 3]
            vuv = sample[:, -4]
            vuv[vuv < 0.5] = 0.0
            vuv[vuv >= 0.5] = 1.0
            bap_full = sample[:, -3:]

            if apply_mlpg:
                mlpg = MLPG()
                lf0 = mlpg.generation(lf0_full, self.cov_lf0, 1)
                # lf0[vuv == 0] = -1e10
                # lf0[lf0 <= math.log(20)] = -1e10  # Fix for WORLD vocoder which will cause memory error otherwise.
                coded_sp = mlpg.generation(coded_sp_full, self.cov_coded_sp, self.cov_coded_sp.shape[0] // 3)
                bap = mlpg.generation(bap_full, self.cov_bap, self.cov_bap.shape[0] // 3)
            else:
                # Version without MLPG.
                coded_sp = coded_sp_full[:, :self.num_coded_sps]
                lf0 = lf0_full[:, 0:1]
                bap = bap_full[:, 0:1]

            sample = np.concatenate((coded_sp, lf0, vuv[:, None], bap), axis=1)
            # Use the following line to ignore MLPG.
            # sample = np.concatenate((coded_sp_full[:, :WorldFeatLabelGen.num_coded_sps], lf0_full[:, :1], vuv[:, None], bap_full[:, :1]), axis=1)

        return sample

    @staticmethod
    def load_sample(id_name, dir_out, add_deltas=False, num_coded_sps=60):
        """
        Load world features from dir_out.

        :param id_name:         Id of the sample.
        :param dir_out:         Directory containing the sample.
        :param add_deltas:      Determines if deltas and double deltas are expected.
        :param num_coded_sps:   Number of bins used to represent the coded spectral features.
        :return:                Numpy array with dimensions num_frames x len(coded_sp, lf0, vuv, bap).
        """
        id_name = os.path.splitext(os.path.basename(id_name))[0]
        logging.debug("Load WORLD " + ("deltas " if add_deltas else "") + "features for " + id_name)

        deltas_factor = 3 if add_deltas else 1
        dim_coded_sp = num_coded_sps * deltas_factor
        dim_lf0 = 1 * deltas_factor
        dim_vuv = 1
        dim_bap = 1 * deltas_factor

        if add_deltas:
            path = os.path.join(dir_out, WorldFeatLabelGen.dir_deltas + "_{}{}".format(WorldFeatLabelGen.dir_coded_sps, num_coded_sps), id_name + WorldFeatLabelGen.ext_deltas)
            with open(path, 'rb') as f:
                try:
                    cmp = np.fromfile(f, dtype=np.float32)
                    labels = np.reshape(cmp, [-1, dim_coded_sp + dim_lf0 + dim_vuv + dim_bap])
                except ValueError as e:
                    logging.error("Cannot load labels from {}.".format(path))
                    raise e
        else:
            path = os.path.join(dir_out, WorldFeatLabelGen.dir_coded_sps + str(num_coded_sps), id_name + WorldFeatLabelGen.ext_coded_sp)
            with open(path, 'rb') as f:
                try:
                    coded_sp = np.fromfile(f, dtype=np.float32)
                    coded_sp = np.reshape(coded_sp, [-1, num_coded_sps])
                except ValueError as e:
                    logging.error("Cannot load labels from {}.".format(path))
                    raise e

            path = os.path.join(dir_out, WorldFeatLabelGen.dir_lf0, id_name + WorldFeatLabelGen.ext_lf0)
            with open(path, 'rb') as f:
                try:
                    lf0 = np.fromfile(f, dtype=np.float32)
                    lf0 = np.reshape(lf0, [-1, dim_lf0])
                except ValueError as e:
                    logging.error("Cannot load labels from {}.".format(path))
                    raise e

            path = os.path.join(dir_out, WorldFeatLabelGen.dir_vuv, id_name + WorldFeatLabelGen.ext_vuv)
            with open(path, 'rb') as f:
                try:
                    vuv = np.fromfile(f, dtype=np.float32)
                    vuv = np.reshape(vuv, [-1, dim_vuv])
                except ValueError as e:
                    logging.error("Cannot load labels from {}.".format(path))
                    raise e

            path = os.path.join(dir_out, WorldFeatLabelGen.dir_bap, id_name + WorldFeatLabelGen.ext_bap)
            with open(path, 'rb') as f:
                try:
                    bap = np.fromfile(f, dtype=np.float32)
                    bap = np.reshape(bap, [-1, dim_bap])
                except ValueError as e:
                    logging.error("Cannot load labels from {}.".format(path))
                    raise e

            # print(coded_sp.shape)
            # print(lf0.shape)
            # print(vuv.shape)
            # print(bap.shape)

            labels = np.concatenate((coded_sp, lf0, vuv, bap), axis=1)
            # min_len = min(len(coded_sp), len(lf0), len(vuv), len(bap))
            # labels = np.concatenate((coded_sp[:min_len], lf0[:min_len], vuv[:min_len], bap[:min_len]), axis=1)

        return labels

    @staticmethod
    def convert_to_world_features(sample, contains_deltas=False, num_coded_sps=60):
        """Convert world acoustic features w/ or w/o deltas to WORLD understandable features."""

        deltas_factor = 3 if contains_deltas else 1

        assert(sample.shape[1] == (num_coded_sps + 2) * deltas_factor + 1)

        coded_sp = sample[:, :num_coded_sps]
        lf0 = sample[:, num_coded_sps * deltas_factor]
        vuv = np.copy(sample[:, num_coded_sps * deltas_factor + deltas_factor])
        vuv[vuv < 0.5] = 0.0
        vuv[vuv >= 0.5] = 1.0
        bap = sample[:, -deltas_factor]

        return coded_sp, lf0, vuv, bap

    @staticmethod
    def mgc_to_sp(mgc, synth_fs):
        fft_size = pyworld.get_cheaptrick_fft_size(synth_fs)
        ln_sp = pysptk.mgc2sp(np.ascontiguousarray(mgc, dtype=np.float64), alpha=WorldFeatLabelGen.mgc_alpha, gamma=0.0, fftlen=fft_size)
        return ln_sp

    def get_normalisation_params(self, dir_out, file_name=None):
        """
        Read the mean std_dev values from a file.
        Save them in self.norm_params.

        :param dir_out:       Directory containing the normalisation file.
        :param file_name:     Prefix of normalisation file.
                              Expects file to be named <file_name-><MeanStdDevExtractor.file_name_appendix>.bin
        :return:              Tuple of normalisation parameters (mean, std_dev).
        """

        if not self.add_deltas:
            # Collect all means and std_devs in a list.
            all_mean = list()
            all_std_dev = list()
            full_file_name = (file_name + "-" if file_name is not None else "") + MeanStdDevExtractor.file_name_appendix + ".bin"
            # Load normalisation parameters for all features.
            for dir_feature in [self.dir_coded_sps, self.dir_lf0, self.dir_bap]:
                mean, std_dev = MeanStdDevExtractor.load(os.path.join(dir_out, dir_feature, full_file_name))
                all_mean.append(np.atleast_2d(mean))
                all_std_dev.append(np.atleast_2d(std_dev))
            # Manually set vuv normalisation paramters.
            all_mean.insert(-1, np.atleast_2d(0.0))
            all_std_dev.insert(-1, np.atleast_2d(1.0))

            # Save the concatenated normalisation parameters locally.
            self.norm_params = np.concatenate(all_mean, axis=1), np.concatenate(all_std_dev, axis=1)
        else:
            full_file_name = (file_name + "-" if file_name is not None else "") + MeanCovarianceExtractor.file_name_appendix # + "_" + self.dir_coded_sps + ".bin"
            # Load the normalisation parameters.
            mean_coded_sp, self.cov_coded_sp, std_dev_coded_sp = MeanCovarianceExtractor.load(os.path.join(dir_out, self.dir_deltas, full_file_name + "_" + self.dir_coded_sps + ".bin"))
            mean_lf0, self.cov_lf0, std_dev_lf0 = MeanCovarianceExtractor.load(os.path.join(dir_out, self.dir_deltas, full_file_name + "_" + self.dir_lf0 + ".bin"))
            mean_bap, self.cov_bap, std_dev_bap = MeanCovarianceExtractor.load(os.path.join(dir_out, self.dir_deltas, full_file_name + "_" + self.dir_bap + ".bin"))

            self.norm_params = (np.concatenate((mean_coded_sp, mean_lf0, np.atleast_1d(0.0), mean_bap)), np.concatenate((std_dev_coded_sp, std_dev_lf0, np.atleast_1d(1.0), std_dev_bap)))

        return self.norm_params

    def gen_data(self, dir_in, dir_out=None, file_id_list=None, id_list=None, add_deltas=False, return_dict=False):
        """
        Prepare WORLD features from audio files. If add_delta is false labels have the dimension
        num_frames x (num_coded_sps + 3) [mgc(num_coded_sps), lf0, vuv, bap(1)], otherwise
        the deltas and double deltas are added between the features resulting in
        num_frames x (3*num_coded_sps + 7) [mgc(3*num_coded_sps), lf0(3*1), vuv, bap(3*1)].

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
                makedirs_safe(os.path.join(dir_out, self.dir_deltas))
            else:
                makedirs_safe(os.path.join(dir_out, self.dir_lf0))
                makedirs_safe(os.path.join(dir_out, self.dir_vuv))
                makedirs_safe(os.path.join(dir_out, self.dir_coded_sps))
                makedirs_safe(os.path.join(dir_out, self.dir_bap))

        # Create the return dictionary if required.
        if return_dict:
            label_dict = OrderedDict()

        if add_deltas:
            # Create normalisation computation units.
            norm_params_ext_coded_sp = MeanCovarianceExtractor()
            norm_params_ext_lf0 = MeanCovarianceExtractor()
            norm_params_ext_bap = MeanCovarianceExtractor()
        else:
            # Create normalisation computation units.
            norm_params_ext_coded_sp = MeanStdDevExtractor()
            norm_params_ext_lf0 = MeanStdDevExtractor()
            # norm_params_ext_vuv = MeanStdDevExtractor()
            norm_params_ext_bap = MeanStdDevExtractor()

        logging.info("Extract WORLD{} features for".format("" if not add_deltas else " deltas")
                     + "[{0}]".format(", ".join(str(i) for i in id_list)))
        for file_name in id_list:

            # Load audio file and extract features.
            audio_name = os.path.join(dir_in, file_name + ".wav")
            raw, fs = soundfile.read(audio_name)
            logging.debug("Extract WORLD{} features from {} at {}Hz.".format("" if not add_deltas else " deltas", file_name, fs))
            f0, sp, ap = pyworld.wav2world(raw, fs)

            file_name = os.path.basename(file_name)  # Remove speaker.

            # Compute lf0 and vuv information.
            lf0 = np.log(f0.clip(min=1E-10), dtype=np.float32)
            lf0[lf0 <= math.log(self.f0_silence_threshold)] = self.lf0_zero
            lf0, vuv = interpolate_lin(lf0)
            lf0 = lf0.astype(dtype=np.float32)
            vuv = vuv.astype(dtype=np.float32)
            # Throw a warning when less then 5% of all frames are unvoiced.
            if vuv.sum() / len(vuv) < 0.05:
                self.logger.warning("Detected only {:.0f}% [{}/{}] unvoiced frames in {}.".format(vuv.sum() / len(vuv) * 100.0, int(vuv.sum()), len(vuv), file_name))

            # Decode spectrum to a lower dimension and aperiodicity to one band aperiodicity.
            # coded_sp = pyworld.code_spectral_envelope(sp, fs, WorldFeatLabelGen.num_coded_sps)  # Cepstral version.
            coded_sp = np.sqrt(sp) * 32768.0
            coded_sp = np.array(pysptk.mcep(coded_sp,
                                            order=self.num_coded_sps - 1,
                                            alpha=self.mgc_alpha,
                                            eps=1.0e-8,
                                            min_det=0.0,
                                            etype=1,
                                            itype=3),
                                dtype=np.float32)
            bap = np.array(pyworld.code_aperiodicity(ap, fs), dtype=np.float32)

            if add_deltas:
                # Compute the deltas and double deltas for all features.
                lf0_deltas, lf0_double_deltas = compute_deltas(lf0)
                coded_sp_deltas, coded_sp_double_deltas = compute_deltas(coded_sp)
                bap_deltas, bap_double_deltas = compute_deltas(bap)

                coded_sp = np.concatenate((coded_sp, coded_sp_deltas, coded_sp_double_deltas), axis=1)
                lf0 = np.concatenate((lf0, lf0_deltas, lf0_double_deltas), axis=1)
                bap = np.concatenate((bap, bap_deltas, bap_double_deltas), axis=1)

                # Combine them to a single feature sample.
                labels = np.concatenate((coded_sp, lf0, vuv, bap), axis=1)

                # Save into return dictionary and/or file.
                if return_dict:
                    label_dict[file_name] = labels
                if dir_out is not None:
                    labels.tofile(os.path.join(dir_out, self.dir_deltas, file_name + self.ext_deltas))

            else:
                # Save into return dictionary and/or file.
                if return_dict:
                    label_dict[file_name] = np.concatenate((coded_sp, lf0, vuv, bap), axis=1)
                if dir_out is not None:
                    coded_sp.tofile(os.path.join(dir_out, self.dir_coded_sps, file_name + self.ext_coded_sp))
                    lf0.tofile(os.path.join(dir_out, self.dir_lf0, file_name + self.ext_lf0))
                    vuv.astype(np.float32).tofile(
                        os.path.join(dir_out, self.dir_vuv, file_name + self.ext_vuv))
                    bap.tofile(os.path.join(dir_out, self.dir_bap, file_name + self.ext_bap))

            # Add sample to normalisation computation unit.
            norm_params_ext_coded_sp.add_sample(coded_sp)
            norm_params_ext_lf0.add_sample(lf0)
            # norm_params_ext_vuv.add_sample(vuv)
            norm_params_ext_bap.add_sample(bap)

        # Save mean and std dev of all features.
        if not add_deltas:
            norm_params_ext_coded_sp.save(os.path.join(dir_out, self.dir_coded_sps, file_id_list_name))
            norm_params_ext_lf0.save(os.path.join(dir_out, self.dir_lf0, file_id_list_name))
            # norm_params_ext_vuv.save(os.path.join(dir_out, WorldFeatLabelGen.dir_vuv, file_id_list_name))
            norm_params_ext_bap.save(os.path.join(dir_out, self.dir_bap, file_id_list_name))
        else:
            self.logger.info("Write norm_prams to{}".format(os.path.join(dir_out, self.dir_deltas, "_".join((file_id_list_name, self.dir_coded_sps)))))
            norm_params_ext_coded_sp.save(os.path.join(dir_out, self.dir_deltas, "_".join((file_id_list_name, self.dir_coded_sps))))
            norm_params_ext_lf0.save(os.path.join(dir_out, self.dir_deltas, "_".join((file_id_list_name, self.dir_lf0))))
            norm_params_ext_bap.save(os.path.join(dir_out, self.dir_deltas, "_".join((file_id_list_name, self.dir_bap))))

        # Get normalisation parameters.
        if not add_deltas:
            norm_coded_sp = norm_params_ext_coded_sp.get_params()
            norm_lf0 = norm_params_ext_lf0.get_params()
            # norm_vuv = norm_params_ext_vuv.get_params()
            norm_bap = norm_params_ext_bap.get_params()

            norm_first = np.concatenate((norm_coded_sp[0], norm_lf0[0], (0.0,), norm_bap[0]), axis=0)
            norm_second = np.concatenate((norm_coded_sp[1], norm_lf0[1], (1.0,), norm_bap[1]), axis=0)

        else:
            norm_coded_sp = norm_params_ext_coded_sp.get_params()
            norm_lf0 = norm_params_ext_lf0.get_params()
            # norm_vuv = norm_params_ext_vuv.get_params()
            norm_bap = norm_params_ext_bap.get_params()

            norm_first = (norm_coded_sp[0], norm_lf0[0], (0.0,), norm_bap[0])
            norm_second = (norm_coded_sp[1], norm_lf0[1], (1.0,), norm_bap[1])

        if return_dict:
            # Return dict of labels for all utterances.
            return label_dict, norm_first, norm_second
        else:
            return norm_first, norm_second


def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-a", "--dir_audio", help="Directory containing the audio (wav) files.",
                        type=str, dest="dir_audio", required=True)
    parser.add_argument("-n", "--num_coded_sps", help="Dimension of the frequency representation.",
                        type=int, dest="num_coded_sps", default=60)
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
    world_feat_gen = WorldFeatLabelGen(dir_out, add_deltas=args.add_deltas, num_coded_sps=args.num_coded_sps)
    world_feat_gen.gen_data(dir_audio,
                            dir_out=dir_out,
                            file_id_list=args.file_id_list_path,
                            id_list=id_list,
                            add_deltas=args.add_deltas,
                            return_dict=False)

    # # DEBUG
    # label_dict, norm_first, norm_second = world_feat_gen.gen_dict(dir_audio,
    #                                                               dir_out=dir_out,
    #                                                               file_id_list=args.file_id_list_path,
    #                                                               id_list=id_list,
    #                                                               add_deltas=args.add_deltas,
    #                                                               return_dict=True)
    #
    # # Call this once before starting the preprocessing.
    # world_feat_gen.get_normalisation_params(dir_out, file_name=file_id_list_name)
    # test_label = label_dict["roger_5535"]
    # print(test_label[98:102, (179 if args.add_deltas else 59):])
    # test_label = world_feat_gen.preprocess_sample(test_label)
    # print(test_label[98:102, (179 if args.add_deltas else 59):])
    # test_label = world_feat_gen.postprocess_sample(test_label)
    # print(test_label[98:102, (179 if args.add_deltas else 59):])

    sys.exit(0)


if __name__ == "__main__":
    main()
