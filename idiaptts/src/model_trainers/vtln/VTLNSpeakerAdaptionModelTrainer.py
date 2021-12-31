#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#

"""Module description:
   Train a model to generate MGC, LF0, VUV, and BAP.
   The model consists of any pre-network which predicts the same features and a warping layer,
   which warps the MGC features based on an internal warping factor prediction. The whole layer is back-propagatable.
"""

# System imports.
import copy
import logging

import numpy as np
import os
import itertools

# Third-party imports.
import librosa
from nnmnkwii import metrics

# Local source tree imports.
from idiaptts.misc.utils import interpolate_lin
from idiaptts.src.Metrics import Metrics
from idiaptts.src.model_trainers.AcousticModelTrainer import AcousticModelTrainer
from idiaptts.src.data_preparation.world.WorldFeatLabelGen import WorldFeatLabelGen
from idiaptts.src.data_preparation.questions.QuestionLabelGen import QuestionLabelGen
from idiaptts.src.DataPlotter import DataPlotter
from idiaptts.src.neural_networks.pytorch.layers.AllPassWarpLayer import AllPassWarpLayer
from idiaptts.src.neural_networks.pytorch.models.AllPassWarpModel import AllPassWarpModel
from torch.nn.modules.module import ModuleAttributeError


class VTLNSpeakerAdaptionModelTrainer(AcousticModelTrainer):

    logger = logging.getLogger(__name__)

    def _get_dummy_warping_layer(self, hparams, mean=None, std_dev=None):
        """Create a model for manual warping."""
        hparams = copy.deepcopy(hparams)
        hparams.add_deltas = False

        config = AllPassWarpLayer.Config(alpha_layer_in_dims=None, alpha_ranges=None, batch_first=True,
                                         warp_matrix_size=hparams.num_coded_sps, mean=mean, std_dev=std_dev)
        transform = config.create_model()
        if hparams.use_gpu:
            transform = transform.cuda()

        return transform

    def compute_score(self, data, output, hparams):
        metrics_dict = super().compute_score(data, output, hparams)

        dict_original_post = self.get_output_dict(
            data.keys(), hparams,
            chunk_size=hparams.n_frames_per_step if hparams.has_value("n_frames_per_step") else 1)

        # Create a warping layer for manual warping.
        wl = self._get_dummy_warping_layer(hparams)
        # norm_params_no_deltas = (self.OutputGen.norm_params[0][:hparams.num_coded_sps],
        #                          self.OutputGen.norm_params[1][:hparams.num_coded_sps])

        # Compute MCD for different set of coefficients.
        batch_size = len(data.keys())
        for cep_coef_start in [1]:
            for cep_coef_end in itertools.chain(range(10, 19), [-1]):
                org_to_output_mcd = 0.0
                org_to_pre_net_output_mcd = 0.0
                for label_name in next(iter(data.values())).keys():
                    for id_name, labels in data.items():
                        labels = labels[label_name]
                        # alphas = output[id_name]['alphas']
                        alphas = [output[id_name][key] for key in output[id_name].keys() if "alphas" in key]
        # batch_size = len(dict_outputs_post)
        # for cep_coef_start in [1]:
        #     for cep_coef_end in itertools.chain(range(10, 19), [-1]):

        #         for id_name, labels in dict_outputs_post.items():
        #             # Split NN output.
        #             _, (output_alphas,) = dict_hiddens[id_name]
                        output_mgc_post, *_ = WorldFeatLabelGen.convert_to_world_features(
                            sample=labels,
                            contains_deltas=False,
                            num_coded_sps=hparams.num_coded_sps,
                            num_bap=hparams.num_bap)
                        # Reverse the warping.
                        pre_net_output, _ = wl.forward_sample(labels[:, :hparams.num_coded_sps], [-alpha for alpha in alphas])
                        # pre_net_output, _ = wl.forward_sample(labels, -output_alphas)
                        pre_net_output = pre_net_output.detach().cpu().numpy()
                        pre_net_mgc = pre_net_output[0, :, :hparams.num_coded_sps]  # * norm_params_no_deltas[1] + norm_params_no_deltas[0]
                        # Load the original warped sample.
                        org_mgc_post = dict_original_post[id_name][:len(output_mgc_post), :hparams.num_coded_sps]

                        # Compute mcd difference.
                        org_to_output_mcd += Metrics.mcd_k(org_mgc_post, output_mgc_post, k=cep_coef_end,
                                                           start_bin=cep_coef_start)
                        org_to_pre_net_output_mcd += Metrics.mcd_k(org_mgc_post, pre_net_mgc, k=cep_coef_end,
                                                                   start_bin=cep_coef_start)
                org_to_pre_net_output_mcd /= batch_size
                org_to_output_mcd /= batch_size

                self.logger.info("MCep from {} to {}:".format(cep_coef_start, cep_coef_end))
                self.logger.info("Original mgc to pre-net mgc error: {:4.2f}dB".format(org_to_pre_net_output_mcd))
                self.logger.info("Original mgc to nn mgc error: {:4.2f}dB".format(org_to_output_mcd))

        return metrics_dict
