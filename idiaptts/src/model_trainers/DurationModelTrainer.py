#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#

"""Module description:
   Train a model to predict durations for phonemes.
"""

# System imports.
import logging
import numpy as np

# Third-party imports.

# Local source tree imports.
from idiaptts.src.data_preparation.phonemes.PhonemeDurationLabelGen import PhonemeDurationLabelGen
from idiaptts.src.Metrics import Metrics
from idiaptts.src.model_trainers.ModularTrainer import ModularTrainer


class DurationModelTrainer(ModularTrainer):
    """
    Implementation of a ModularTrainer for the generation of durations
    for phonemes. Use phonemes (indices or one-hot) as input and predict
    durations.
    """
    logger = logging.getLogger(__name__)

    @staticmethod
    def create_hparams(hparams_string=None, verbose=False):
        hparams = ModularTrainer.create_hparams(hparams_string, verbose=False)
        hparams.add_hparams(  # exclude_begin_and_end_silence=False,
                            # htk_min_phoneme_length=50000,
                            # phoneme_label_type="HTK full",  # Specifies the format in which the .lab files are stored.
                            #                                 # Refer to PhonemeLabelGen.load_sample for a list of types.
                            metrics=[Metrics.Dur_RMSE, Metrics.Dur_pearson])

        if verbose:
            logging.info(hparams.get_debug_string())

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
        raise NotImplementedError()
        output_dict, output_dict_post = super().forward(hparams, id_list)  # Call base class forward.

        # Convert output into phoneme lengths.
        output_dict_post.update((key, np.around(value).astype(np.int) * hparams.htk_min_phoneme_length) for key, value in output_dict_post.items())

        # Ensure positivity of predicted durations if requested.
        if only_positive:
            for output in output_dict_post.values():
                output[output < 0] = 0

        return output_dict, output_dict_post

    def compute_score(self, data, output, hparams):

        dict_original_post = self.get_output_dict(data.keys(), hparams)

        metric_dict = {}
        for label_name in next(iter(data.values())).keys():
            metric = Metrics(hparams.metrics)
            for id_name, labels in data.items():
                labels = labels[label_name]
                org_dur = dict_original_post[id_name]

                current_metrics = metric.get_metrics(hparams.metrics,
                                                     org_dur=org_dur,
                                                     output_dur=labels)
                metric.accumulate(id_name, current_metrics)

            metric.log()
            metric_dict[label_name] = metric.get_cum_values()

        return metric_dict

    def get_output_dict(self, id_list, hparams):
        assert hparams.has_value("dur_dir"), \
            "hparams.dur_dir must be set for this operation."
        dict_original_post = dict()
        for id_name in id_list:
            dict_original_post[id_name] = PhonemeDurationLabelGen.load_sample(
                id_name, hparams.dur_dir)

        return dict_original_post

    def gen_figure_from_output(self, id_name, output, hidden, hparams):
        # TODO: Is there a reasonable way to plot duration prediction?
        NotImplementedError()
