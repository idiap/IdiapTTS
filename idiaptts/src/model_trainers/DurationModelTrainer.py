#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#

"""Module description:
   Train a model to predict durations for phoneme states.
"""

# System imports.
import logging
import math
import numpy as np
import os
import scipy

# Third-party imports.
import torch

# Local source tree imports.
from idiaptts.src.model_trainers.ModelTrainer import ModelTrainer
from idiaptts.src.data_preparation.phonemes.PhonemeLabelGen import PhonemeLabelGen
from idiaptts.src.data_preparation.phonemes.PhonemeDurationLabelGen import PhonemeDurationLabelGen
from idiaptts.src.data_preparation.PyTorchLabelGensDataset import PyTorchLabelGensDataset


class DurationModelTrainer(ModelTrainer):
    """
    Implementation of a ModelTrainer for the generation of durations for phonemes.

    Use phonemes as input and predict durations for five states.
    """
    logger = logging.getLogger(__name__)

    #########################
    # Default constructor
    #
    def __init__(self, dir_phoneme_labels, dir_durations, id_list, file_symbol_dict, hparams=None):
        """Default constructor.

        :param dir_phoneme_labels:      Path to the directory containing the label files with monophones.
        :param dir_durations:           Path to the directory containing the durations.
        :param id_list:                 List containing all ids. Subset is taken as test set.
        :param file_symbol_dict:        List of all used monophones.
        """
        if hparams is None:
            hparams = self.create_hparams()
            hparams.out_dir = os.path.curdir

        # Write missing default parameters.
        if hparams.variable_sequence_length_train is None:
            hparams.variable_sequence_length_train = hparams.batch_size_train > 1
        if hparams.variable_sequence_length_test is None:
            hparams.variable_sequence_length_test = hparams.batch_size_test > 1
        if not hasattr(hparams, "synth_dir") or hparams.synth_dir is None:
            hparams.synth_dir = os.path.join(hparams.out_dir, "synth")

        super().__init__(id_list, hparams)

        self.InputGen = PhonemeLabelGen(dir_phoneme_labels, file_symbol_dict, hparams.phoneme_label_type, one_hot=True)
        self.OutputGen = PhonemeDurationLabelGen(dir_durations)
        self.OutputGen.get_normalisation_params(dir_durations, hparams.output_norm_params_file_prefix)

        self.dataset_train = PyTorchLabelGensDataset(self.id_list_train, self.InputGen, self.OutputGen, hparams, match_lengths=False)
        self.dataset_val = PyTorchLabelGensDataset(self.id_list_val, self.InputGen, self.OutputGen, hparams, match_lengths=False)

        if self.loss_function is None:
            self.loss_function = torch.nn.MSELoss(reduction='none')

        if hparams.scheduler_type == "default":
            hparams.scheduler_type = "Plateau"
            hparams.add_hparams(plateau_verbose=True)

    @staticmethod
    def create_hparams(hparams_string=None, verbose=False):
        hparams = ModelTrainer.create_hparams(hparams_string, verbose=False)
        hparams.add_hparams(  # exclude_begin_and_end_silence=False,
                            phoneme_label_type="HTK full")  # Specifies the format in which the .lab files are stored.
                                                            # Refer to PhonemeLabelGen.load_sample for a list of possible types.

        if verbose:
            logging.info('Final parsed hparams: %s', hparams.values())

        return hparams

    def forward(self, hparams, id_list, only_positive=True):
        """
        Forward all given ids through the network in batches of hparams.batch_size_val and converts the output to
        multiples of hparams.min_phoneme_length.

        :param hparams:        Hyper-parameter container.
        :param id_list:        Can be full path to file with ids, list of ids, or one id.
        :param only_positive:  If True, sets all negative values in the post-processed dictionary to 0.
        :return:               (Dictionary of network outputs, dictionary of post-processed (by self.OutputGen) network outputs)
        """
        output_dict, output_dict_post = super().forward(hparams, id_list)  # Call base class forward.

        # Convert output into phoneme lengths.
        output_dict_post.update((key, np.around(value).astype(np.int) * hparams.min_phoneme_length) for key, value in output_dict_post.items())

        # Ensure positivity of predicted durations if requested.
        if only_positive:
            for output in output_dict_post.values():
                output[output < 0] = 0

        return output_dict, output_dict_post

    def compute_score(self, dict_outputs_post, dict_hiddens, hparams):

        # Get data for comparision.
        dict_original_post = dict()
        for id_name in dict_outputs_post.keys():
            dict_original_post[id_name] = PhonemeDurationLabelGen.load_sample(id_name, self.OutputGen.dir_labels)

        rmse = 0.0
        rmse_max_id = "None"
        rmse_max = 0.0
        all_rmse = []
        pearsonr = np.zeros(next(iter(dict_original_post.values())).shape[1], dtype=np.float32)

        for id_name, output_dur in dict_outputs_post.items():
            org_dur = dict_original_post[id_name]

            # Compute RMSE.
            mse = (org_dur - output_dur) ** 2
            current_rmse = math.sqrt(mse.sum())
            if current_rmse > rmse_max:
                rmse_max_id = id_name
                rmse_max = current_rmse
            rmse += current_rmse
            all_rmse.append(current_rmse)

            # Compute pearson correlation.
            for idx in range(org_dur.shape[1]):
                pearsonr[idx] += scipy.stats.pearsonr(org_dur[:, idx], output_dur[:, idx])[0]

        rmse /= len(dict_outputs_post)
        pearsonr /= len(dict_original_post)

        self.logger.info("Worst RMSE: {} {:4.2f}".format(rmse_max_id, rmse_max))
        self.logger.info("Duration RMSE {:4.2f}Hz, Pearson correlation {}.".format(rmse, np.array_str(pearsonr, precision=2, suppress_small=True)))

        return rmse

    def gen_figure_from_output(self, id_name, output, hidden, hparams):
        # Is there a reasonable way to plot duration prediction?
        pass
