#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#

"""Module description:
   Train a model to predict atoms and generate LF0 from them.
   Format is T x (|thetas| + 1) with one amplitude per theta and a position flag at last position.
   Each amplitude in the target labels is surrounded by a distribution.
   Combine LF0 data with external MGC and BAP data to synthesize audio.
"""

# System imports.
import logging
import math
import os
import sys
import numpy as np

# Third-party imports.

# Local source tree imports.
from idiaptts.src.model_trainers.ModelTrainer import ModelTrainer
from idiaptts.src.model_trainers.wcad.AtomModelTrainer import AtomModelTrainer
from idiaptts.src.data_preparation.questions.QuestionLabelGen import QuestionLabelGen
from idiaptts.src.data_preparation.wcad.AtomVUVDistPosLabelGen import AtomVUVDistPosLabelGen
from idiaptts.src.data_preparation.PyTorchLabelGensDataset import PyTorchLabelGensDataset
from idiaptts.src.data_preparation.world.LF0LabelGen import LF0LabelGen
from idiaptts.src.data_preparation.wcad.AtomLabelGen import AtomLabelGen
from idiaptts.src.data_preparation.world.WorldFeatLabelGen import WorldFeatLabelGen
from idiaptts.src.neural_networks.pytorch.loss.WeightedNonzeroWMSEAtomLoss import WeightedNonzeroWMSEAtomLoss
from idiaptts.src.DataPlotter import DataPlotter
from idiaptts.misc.utils import interpolate_lin


class AtomVUVDistPosModelTrainer(AtomModelTrainer):
    """
    Subclass of AtomModelTrainer, which uses one amplitude per theta plus position flag,
    format is T x (|thetas| + 1). Each amplitude in the target labels is surrounded by a distribution.
    Positions of atoms are identified by finding the peaks of the position flag prediction. For positive peaks
    the theta with the highest amplitude is used, for negative peaks the theta with the lowest amplitude.
    Acoustic data is generated from these atoms. MGC and BAP is either generated by a pre-trained acoustic model
    or loaded from the original extracted files. Question labels are used as input.
    """
    logger = logging.getLogger(__name__)

    def __init__(self, wcad_root, dir_atom_labels, dir_lf0_labels, dir_question_labels, id_list,
                 thetas, k,
                 num_questions,
                 dist_window_size=51, hparams=None):
        """Default constructor.

        :param wcad_root:               Path to main directory of wcad.
        :param dir_atom_labels:         Path to directory that contains the .wav files.
        :param dir_lf0_labels:          Path to directory that contains the .lf0 files.
        :param dir_question_labels:     Path to directory that contains the .lab files.
        :param id_list:                 List containing all ids. Subset is taken as test set.
        :param thetas:                  List of theta values of atoms.
        :param k:                       K-value of atoms.
        :param num_questions:           Expected number of questions in question labels.
        :param dist_window_size:        Width of the distribution surrounding each atom spike
                                        The window is only used for amps. Thetas are surrounded by a window of 5.
        :param hparams:                 Hyper-parameter container.
        """
        if hparams is None:
            hparams = self.create_hparams()
            hparams.out_dir = os.path.curdir

        # Write missing default parameters.
        if hparams.synth_dir is None:
            hparams.synth_dir = os.path.join(hparams.out_dir, "synth")

        # If the weight for unvoiced frames is not given, compute it to get equal weights.
        if not hasattr(hparams, "weight_zero") or hparams.weight_zero is None:
            non_zero_occurrence = min(0.99, 0.015 / len(thetas))
            zero_occurrence = 1 - non_zero_occurrence
            hparams.add_hparam("weight_non_zero", 1 / non_zero_occurrence)
            hparams.add_hparam("weight_zero", 1 / zero_occurrence)
        if not hasattr(hparams, "weight_vuv") or hparams.weight_vuv is None:
            hparams.add_hparam("weight_vuv", 0.5)
        if not hasattr(hparams, "atom_loss_theta") or hparams.atom_loss_theta is None:
            hparams.add_hparam("atom_loss_theta", 0.01)

        # Explicitly call only the constructor of the baseclass of AtomModelTrainer.
        super(AtomModelTrainer, self).__init__(id_list, hparams)

        if hparams.dist_window_size % 2 == 0:
            hparams.dist_window_size += 1
            self.logger.warning("hparams.dist_window_size should be odd, changed it to " + str(hparams.dist_window_size))

        self.InputGen = QuestionLabelGen(dir_question_labels, num_questions)
        self.InputGen.get_normalisation_params(dir_question_labels, hparams.input_norm_params_file_prefix)

        # Overwrite OutputGen by the one with beta distribution.
        self.OutputGen = AtomVUVDistPosLabelGen(wcad_root, dir_atom_labels, dir_lf0_labels, thetas, k, hparams.frame_size_ms, window_size=dist_window_size)
        self.OutputGen.get_normalisation_params(dir_atom_labels, hparams.output_norm_params_file_prefix)

        self.dataset_train = PyTorchLabelGensDataset(self.id_list_train, self.InputGen, self.OutputGen, hparams, match_lengths=True)
        self.dataset_val = PyTorchLabelGensDataset(self.id_list_val, self.InputGen, self.OutputGen, hparams, match_lengths=True)

        if self.loss_function is None:
            self.loss_function = WeightedNonzeroWMSEAtomLoss(use_gpu=hparams.use_gpu,
                                                             theta=hparams.atom_loss_theta,
                                                             weights_vuv=hparams.weight_vuv,
                                                             weights_zero=hparams.weight_zero,
                                                             weights_non_zero=hparams.weight_non_zero,
                                                             reduce=False)

        if hparams.scheduler_type == "default":
            hparams.scheduler_type = "None"

    @staticmethod
    def create_hparams(hparams_string=None, verbose=False):
        hparams = AtomModelTrainer.create_hparams(hparams_string, verbose=False)
        hparams.add_hparam("dist_window_size", 51)
        hparams.add_hparam("synth_acoustic_model", None)

        if verbose:
            logging.info(hparams.get_debug_string())

        return hparams

    def gen_figure_from_output(self, id_name, label, hidden, hparams):

        # Retrieve data from label.
        output_amps = label[:, 1:-1]
        output_pos = label[:, -1]
        labels_post = self.OutputGen.postprocess_sample(label)
        output_vuv = labels_post[:, 0, 1].astype(bool)
        output_atoms = self.OutputGen.labels_to_atoms(labels_post, k=hparams.k, amp_threshold=hparams.min_atom_amp)
        output_lf0 = self.OutputGen.atoms_to_lf0(output_atoms, len(label))

        # Load original lf0 and vuv.
        world_dir = hparams.world_dir if hasattr(hparams, "world_dir") and hparams.world_dir is not None\
                                      else os.path.join(self.OutputGen.dir_labels, self.dir_extracted_acoustic_features)
        org_labels = LF0LabelGen.load_sample(id_name, world_dir)
        original_lf0, _ = LF0LabelGen.convert_to_world_features(org_labels)
        original_lf0, _ = interpolate_lin(original_lf0)

        phrase_curve = np.fromfile(os.path.join(self.OutputGen.dir_labels, id_name + self.OutputGen.ext_phrase),
                                   dtype=np.float32).reshape(-1, 1)
        original_lf0[:len(phrase_curve)] -= phrase_curve[:len(original_lf0)]
        original_lf0 = original_lf0[:len(output_lf0)]

        org_labels = self.OutputGen.load_sample(id_name,
                                                self.OutputGen.dir_labels,
                                                len(hparams.thetas),
                                                self.OutputGen.dir_world_labels)
        org_vuv = org_labels[:, 0, 0].astype(bool)
        org_labels = org_labels[:, 1:]
        len_diff = len(org_labels) - len(labels_post)
        org_labels = self.OutputGen.trim_end_sample(org_labels, int(len_diff / 2.0))
        org_labels = self.OutputGen.trim_end_sample(org_labels, int(len_diff / 2.0) + 1)
        org_atoms = AtomLabelGen.labels_to_atoms(org_labels, k=hparams.k, frame_size=hparams.frame_size_ms)
        wcad_lf0 = self.OutputGen.atoms_to_lf0(org_atoms, len(org_labels))

        # Get a data plotter
        net_name = os.path.basename(hparams.model_name)
        filename = str(os.path.join(hparams.out_dir, id_name + '.' + net_name))
        plotter = DataPlotter()
        plotter.set_title(id_name + " - " + net_name)

        grid_idx = 0
        graphs_output = list()
        for idx in reversed(range(output_amps.shape[1])):
            graphs_output.append((output_amps[:, idx], r'$\theta$={0:.3f}'.format(hparams.thetas[idx])))
        plotter.set_data_list(grid_idx=grid_idx, data_list=graphs_output)
        plotter.set_label(grid_idx=grid_idx, ylabel='NN amps')
        amp_max = np.max(output_amps) * 1.1
        amp_min = np.min(output_amps) * 1.1
        plotter.set_lim(grid_idx=grid_idx, ymin=amp_min, ymax=amp_max)

        grid_idx += 1
        graphs_pos_flag = list()
        graphs_pos_flag.append((output_pos,))
        plotter.set_data_list(grid_idx=grid_idx, data_list=graphs_pos_flag)
        plotter.set_label(grid_idx=grid_idx, ylabel='NN pos')

        grid_idx += 1
        graphs_peaks = list()
        for idx in reversed(range(label.shape[1] - 2)):
            graphs_peaks.append((labels_post[:, 1 + idx, 0],))
        plotter.set_data_list(grid_idx=grid_idx, data_list=graphs_peaks)
        plotter.set_area_list(grid_idx=grid_idx, area_list=[(np.invert(output_vuv), '0.75', 1.0, 'Unvoiced')])
        plotter.set_label(grid_idx=grid_idx, ylabel='NN peaks')
        plotter.set_lim(grid_idx=grid_idx, ymin=-1.8, ymax=1.8)

        grid_idx += 1
        graphs_target = list()
        for idx in reversed(range(org_labels.shape[1])):
            graphs_target.append((org_labels[:, idx, 0],))
        plotter.set_data_list(grid_idx=grid_idx, data_list=graphs_target)
        plotter.set_hatchstyles(grid_idx=grid_idx, hatchstyles=['\\\\'])
        plotter.set_area_list(grid_idx=grid_idx, area_list=[(np.invert(org_vuv.astype(bool)), '0.75', 1.0, 'Reference unvoiced')])
        plotter.set_label(grid_idx=grid_idx, ylabel='target')
        plotter.set_lim(grid_idx=grid_idx, ymin=-1.8, ymax=1.8)

        grid_idx += 1
        graphs_lf0 = list()
        graphs_lf0.append((wcad_lf0, "wcad lf0"))
        graphs_lf0.append((original_lf0, "org lf0"))
        graphs_lf0.append((output_lf0, "predicted lf0"))
        plotter.set_data_list(grid_idx=grid_idx, data_list=graphs_lf0)
        plotter.set_area_list(grid_idx=grid_idx, area_list=[(np.invert(org_vuv.astype(bool)), '0.75', 1.0)])
        plotter.set_hatchstyles(grid_idx=grid_idx, hatchstyles=['\\\\'])
        plotter.set_label(grid_idx=grid_idx, xlabel='frames [' + str(hparams.frame_size_ms) + ' ms]', ylabel='lf0')
        amp_lim = max(np.max(np.abs(wcad_lf0)), np.max(np.abs(output_lf0))) * 1.1
        plotter.set_lim(grid_idx=grid_idx, ymin=-amp_lim, ymax=amp_lim)
        plotter.set_linestyles(grid_idx=grid_idx, linestyles=[':', '--', '-'])

        # # Compute F0 RMSE for sample and add it to title.
        # org_f0 = (np.exp(lf0.squeeze() + phrase_curve[:len(lf0)].squeeze()) * vuv)[:len(output_lf0)]  # Fix minor negligible length mismatch.
        # output_f0 = np.exp(output_lf0 + phrase_curve[:len(output_lf0)].squeeze()) * output_vuv[:len(output_lf0)]
        # f0_mse = (org_f0 - output_f0) ** 2
        # # non_zero_count = np.logical_and(vuv[:len(output_lf0)], output_vuv).sum()
        # f0_rmse = math.sqrt(f0_mse.sum() / (np.logical_and(vuv[:len(output_lf0)], output_vuv).sum()))

        # # Compute vuv error rate.
        # num_errors = (vuv[:len(output_lf0)] != output_vuv)
        # vuv_error_rate = float(num_errors.sum()) / len(output_lf0)
        # plotter.set_title(id_name + " - " + net_name + " - F0_RMSE_" + "{:4.2f}Hz".format(f0_rmse) + " - VUV_" + "{:2.2f}%".format(vuv_error_rate * 100))
        # plotter.set_lim(xmin=300, xmax=1100)g
        plotter.gen_plot(monochrome=True)
        plotter.gen_plot()
        plotter.save_to_file(filename + ".VUV_DIST_POS" + hparams.gen_figure_ext)

    def compute_score(self, dict_outputs_post, dict_hiddens, hparams):
        """Compute the score of a dictionary with post-processes labels."""

        # Get data for comparision.
        dict_original_post = self.load_extracted_audio_features(dict_outputs_post, hparams)

        f0_rmse = 0.0
        vuv_error_rate = 0.0
        f0_rmse_max_id = "None"
        f0_rmse_max = 0.0
        vuv_error_max_id = "None"
        vuv_error_max = 0.0
        for id_name, label in dict_outputs_post.items():

            output_vuv = label[:, 0, 1].astype(bool)
            output_atom_labels = label[:, 1:]
            output_lf0 = self.OutputGen.labels_to_lf0(output_atom_labels, k=hparams.k, frame_size=hparams.frame_size_ms, amp_threshold=hparams.min_atom_amp)

            # Get data for comparision.
            org_lf0 = dict_original_post[id_name][:, hparams.num_coded_sps]
            org_vuv = dict_original_post[id_name][:, hparams.num_coded_sps + 1]
            phrase_curve = self.get_phrase_curve(id_name)

            # Compute f0 from lf0.
            org_f0 = np.exp(org_lf0.squeeze())[:len(output_lf0)]  # Fix minor negligible length mismatch.
            output_f0 = np.exp(output_lf0 + phrase_curve[:len(output_lf0)].squeeze())

            # Compute RMSE, keep track of worst RMSE.
            f0_mse = (org_f0 - output_f0) ** 2
            current_f0_rmse = math.sqrt((f0_mse * org_vuv[:len(output_lf0)]).sum() / org_vuv[:len(output_lf0)].sum())
            if current_f0_rmse > f0_rmse_max:
                f0_rmse_max_id = id_name
                f0_rmse_max = current_f0_rmse
            f0_rmse += current_f0_rmse

            # Compute vuv error rate.
            num_errors = (org_vuv[:len(output_lf0)] != output_vuv)
            vuv_error_rate_tmp = float(num_errors.sum()) / len(output_lf0)
            if vuv_error_rate_tmp > vuv_error_max:
                vuv_error_max_id = id_name
                vuv_error_max = vuv_error_rate_tmp
            vuv_error_rate += vuv_error_rate_tmp

        f0_rmse /= len(dict_outputs_post)
        vuv_error_rate /= len(dict_outputs_post)

        self.logger.info("Worst F0 RMSE: " + f0_rmse_max_id + " {:4.2f}Hz".format(f0_rmse_max))
        self.logger.info("Worst VUV error: " + vuv_error_max_id + " {:2.2f}%".format(vuv_error_max * 100))
        self.logger.info("Benchmark score: F0 RMSE " + "{:4.2f}Hz".format(f0_rmse)
                         + ", VUV " + "{:2.2f}%".format(vuv_error_rate * 100))

        return f0_rmse, vuv_error_rate

    def synthesize(self, id_list, synth_output, hparams):
        """
        Synthesise LF0 from atoms. The run_atom_synth function either loads the original acoustic features or uses an
        acoustic model to predict them.
        """
        full_output = self.run_atom_synth(id_list, synth_output, hparams)

        for id_name, labels in full_output.items():
            lf0 = labels[:, -3]
            lf0, _ = interpolate_lin(lf0)
            vuv = synth_output[id_name][:, 0, 1]
            len_diff = len(labels) - len(vuv)
            labels = WorldFeatLabelGen.trim_end_sample(labels, int(len_diff / 2), reverse=True)
            labels = WorldFeatLabelGen.trim_end_sample(labels, len_diff - int(len_diff / 2))
            labels[:, -2] = vuv

        # Run the vocoder.
        ModelTrainer.synthesize(self, id_list, full_output, hparams)
