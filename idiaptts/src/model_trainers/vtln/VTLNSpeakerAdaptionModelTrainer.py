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
import logging
import numpy as np
import os
import itertools

# Third-party imports.
from nnmnkwii import metrics

# Local source tree imports.
from idiaptts.misc.utils import interpolate_lin
from idiaptts.src.model_trainers.AcousticModelTrainer import AcousticModelTrainer
from idiaptts.src.data_preparation.world.WorldFeatLabelGen import WorldFeatLabelGen
from idiaptts.src.data_preparation.questions.QuestionLabelGen import QuestionLabelGen
from idiaptts.src.DataPlotter import DataPlotter
from idiaptts.src.neural_networks.pytorch.models.WarpingLayer import WarpingLayer


class VTLNSpeakerAdaptionModelTrainer(AcousticModelTrainer):

    logger = logging.getLogger(__name__)

    def init(self, hparams):
        self.logger.info("Create VTLN model.")
        super().init(hparams)
        sp_mean = self.OutputGen.norm_params[0][:hparams.num_coded_sps * (3 if hparams.add_deltas else 1)]
        sp_std_dev = self.OutputGen.norm_params[1][:hparams.num_coded_sps * (3 if hparams.add_deltas else 1)]
        self.model_handler.model.set_norm_params(sp_mean, sp_std_dev)

    @staticmethod
    def create_hparams(hparams_string=None, verbose=False):
        """Create model hyper parameters. Parse non default from given string."""

        hparams = AcousticModelTrainer.create_hparams(hparams_string, verbose=False)
        hparams.add_hparams(
            f_get_emb_index=None,  # Computes an index from the id_name of a sample to index embedding vector.
            pre_net_model_type=None,  # Can be any type registered in ModelFactory.
            pre_net_model_name=None,  # Used to load a model when pre_net_model_type is None.
            pre_net_model_path=None,
            train_pre_net=True,
            pass_embs_to_pre_net=False,
            num_coded_sps=30,
            num_speakers=None,
            speaker_emb_dim=128)

        if verbose:
            logging.info(hparams.get_debug_string())

        return hparams

    def _get_dummy_warping_layer(self, hparams):
        """Create a warping layer for manual warping."""
        prev_add_deltas = hparams.add_deltas
        hparams.add_deltas = False
        wl = WarpingLayer((hparams.num_coded_sps,), (hparams.num_coded_sps,), hparams)
        hparams.add_deltas = prev_add_deltas
        if hparams.use_gpu:
            wl = wl.cuda()
        norm_params_no_deltas = (self.OutputGen.norm_params[0][:hparams.num_coded_sps],
                                 self.OutputGen.norm_params[1][:hparams.num_coded_sps])
        wl.set_norm_params(*norm_params_no_deltas)
        return wl

    def gen_figure_from_output(self, id_name, label, hidden, hparams):
        _, alphas = hidden
        labels_post = self.OutputGen.postprocess_sample(label)
        coded_sp, lf0, vuv, bap = WorldFeatLabelGen.convert_to_world_features(labels_post,
                                                                              contains_deltas=False,
                                                                              num_coded_sps=hparams.num_coded_sps,
                                                                              num_bap=hparams.num_bap)
        sp = WorldFeatLabelGen.mcep_to_amp_sp(coded_sp, hparams.synth_fs)
        lf0, _ = interpolate_lin(lf0)

        # Load original LF0.
        org_labels_post = WorldFeatLabelGen.load_sample(id_name,
                                                        dir_out=self.OutputGen.dir_labels,
                                                        add_deltas=self.OutputGen.add_deltas,
                                                        num_coded_sps=hparams.num_coded_sps,
                                                        num_bap=hparams.num_bap)
        original_mgc, original_lf0, original_vuv, *_ = WorldFeatLabelGen.convert_to_world_features(
                                                                            sample=org_labels_post,
                                                                            contains_deltas=self.OutputGen.add_deltas,
                                                                            num_coded_sps=hparams.num_coded_sps,
                                                                            num_bap=hparams.num_bap)
        original_lf0, _ = interpolate_lin(original_lf0)

        sp = sp[:, :150]  # Zoom into spectral features.

        # Get a data plotter.
        grid_idx = -1
        plotter = DataPlotter()
        net_name = os.path.basename(hparams.model_name)
        filename = str(os.path.join(hparams.out_dir, id_name + '.' + net_name))
        plotter.set_title(id_name + ' - ' + net_name)
        plotter.set_num_colors(3)
        # plotter.set_lim(grid_idx=0, ymin=math.log(60), ymax=math.log(250))

        # # Plot LF0
        # grid_idx += 1
        # graphs.append((original_lf0, 'Original LF0'))
        # graphs.append((lf0, 'NN LF0'))
        # plotter.set_data_list(grid_idx=grid_idx, data_list=graphs)
        # plotter.set_area_list(grid_idx=grid_idx, area_list=[(np.invert(vuv.astype(bool)), '0.8', 1.0),
        #                                                     (np.invert(original_vuv.astype(bool)), 'red', 0.2)])
        # plotter.set_label(grid_idx=grid_idx, xlabel='frames [{}] ms'.format(hparams.frame_length), ylabel='log(f0)')

        # Reverse the warping.
        wl = self._get_dummy_warping_layer(hparams)
        norm_params_no_deltas = (self.OutputGen.norm_params[0][:hparams.num_coded_sps],
                                 self.OutputGen.norm_params[1][:hparams.num_coded_sps])
        pre_net_output, _ = wl.forward_sample(label, -alphas)

        # Postprocess sample manually.
        pre_net_output = pre_net_output.detach().cpu().numpy()
        pre_net_mgc = pre_net_output[:, 0, :hparams.num_coded_sps] * norm_params_no_deltas[1] + norm_params_no_deltas[0]

        # Plot spectral features predicted by pre-network.
        grid_idx += 1
        plotter.set_label(grid_idx=grid_idx,
                          xlabel='frames [{}] ms'.format(hparams.frame_size_ms),
                          ylabel='Pre-network')
        plotter.set_specshow(grid_idx=grid_idx,
                             spec=np.abs(WorldFeatLabelGen.mcep_to_amp_sp(pre_net_mgc, hparams.synth_fs)[:, :sp.shape[1]]))

        # Plot final predicted spectral features.
        grid_idx += 1
        plotter.set_label(grid_idx=grid_idx, xlabel='frames [{}] ms'.format(hparams.frame_size_ms), ylabel='VTLN')
        plotter.set_specshow(grid_idx=grid_idx, spec=np.abs(sp))

        # Plot predicted alpha value and V/UV flag.
        grid_idx += 1
        plotter.set_label(grid_idx=grid_idx, xlabel='frames [{}] ms'.format(hparams.frame_size_ms), ylabel='alpha')
        graphs = list()
        graphs.append((alphas, 'NN alpha'))
        plotter.set_data_list(grid_idx=grid_idx, data_list=graphs)
        plotter.set_area_list(grid_idx=grid_idx, area_list=[(np.invert(vuv.astype(bool)), '0.8', 1.0),
                                                            (np.invert(original_vuv.astype(bool)), 'red', 0.2)])

        # Add phoneme annotations if given.
        if hasattr(hparams, "phoneme_indices") and hparams.phoneme_indices is not None \
           and hasattr(hparams, "question_file") and hparams.question_file is not None:
            questions = QuestionLabelGen.load_sample(id_name,
                                                     os.path.join("experiments", hparams.voice, "questions"),
                                                     num_questions=hparams.num_questions)[:len(lf0)]
            np_phonemes = QuestionLabelGen.questions_to_phonemes(questions,
                                                                 hparams.phoneme_indices,
                                                                 hparams.question_file)
            plotter.set_annotations(grid_idx, np_phonemes)

        # Plot reference spectral features.
        grid_idx += 1
        plotter.set_label(grid_idx=grid_idx,
                          xlabel='frames [{}] ms'.format(hparams.frame_size_ms),
                          ylabel='Original spectrogram')
        plotter.set_specshow(grid_idx=grid_idx,
                             spec=np.abs(WorldFeatLabelGen.mcep_to_amp_sp(original_mgc, hparams.synth_fs)[:, :sp.shape[1]]))

        plotter.gen_plot()
        plotter.save_to_file(filename + '.VTLN' + hparams.gen_figure_ext)

    def compute_score(self, dict_outputs_post, dict_hiddens, hparams):
        mcd, f0_rmse, vuv_error_rate, bap_mcd = super().compute_score(dict_outputs_post, dict_hiddens, hparams)

        # Get data for comparision.
        dict_original_post = dict()
        for id_name in dict_outputs_post.keys():
            dict_original_post[id_name] = WorldFeatLabelGen.load_sample(id_name,
                                                                        dir_out=self.OutputGen.dir_labels,
                                                                        add_deltas=True,
                                                                        num_coded_sps=hparams.num_coded_sps,
                                                                        num_bap=hparams.num_bap)

        # Create a warping layer for manual warping.
        wl = self._get_dummy_warping_layer(hparams)
        norm_params_no_deltas = (self.OutputGen.norm_params[0][:hparams.num_coded_sps],
                                 self.OutputGen.norm_params[1][:hparams.num_coded_sps])

        # Compute MCD for different set of coefficients.
        batch_size = len(dict_outputs_post)
        for cep_coef_start in [1]:
            for cep_coef_end in itertools.chain(range(10, 19), [-1]):
                org_to_output_mcd = 0.0
                org_to_pre_net_output_mcd = 0.0

                for id_name, labels in dict_outputs_post.items():
                    # Split NN output.
                    _, output_alphas = dict_hiddens[id_name]
                    output_mgc_post, *_ = self.OutputGen.convert_to_world_features(
                                                           labels,
                                                           False,
                                                           num_coded_sps=hparams.num_coded_sps,
                                                           num_bap=hparams.num_bap)
                    # Reverse the warping.
                    pre_net_output, _ = wl.forward_sample(labels, -output_alphas)
                    # Postprocess sample manually.
                    pre_net_output = pre_net_output.detach().cpu().numpy()
                    pre_net_mgc = pre_net_output[:, 0, :hparams.num_coded_sps] * norm_params_no_deltas[1] + norm_params_no_deltas[0]
                    # Load the original warped sample.
                    org_mgc_post = dict_original_post[id_name][:len(output_mgc_post), :hparams.num_coded_sps]

                    # Compute mcd difference.
                    org_to_output_mcd += metrics.melcd(org_mgc_post[:, cep_coef_start:cep_coef_end],
                                                       output_mgc_post[:, cep_coef_start:cep_coef_end])
                    org_to_pre_net_output_mcd += metrics.melcd(org_mgc_post[:, cep_coef_start:cep_coef_end],
                                                               pre_net_mgc[:, cep_coef_start:cep_coef_end])

                org_to_pre_net_output_mcd /= batch_size
                org_to_output_mcd /= batch_size

                self.logger.info("MCep from {} to {}:".format(cep_coef_start, cep_coef_end))
                self.logger.info("Original mgc to pre-net mgc error: {:4.2f}dB".format(org_to_pre_net_output_mcd))
                self.logger.info("Original mgc to nn mgc error: {:4.2f}dB".format(org_to_output_mcd))

        return mcd, f0_rmse, vuv_error_rate, bap_mcd
