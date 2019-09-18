#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#

"""Module description:
   Train a model to generate LF0 using end-to-end system including neural filters.
"""


# System imports.
import math
import sys
import logging
import numpy as np
import os
import copy
import torch

# Local source tree imports.
from idiaptts.src.model_trainers.ModelTrainer import ModelTrainer
from idiaptts.src.neural_networks.pytorch.ModelHandlerPyTorch import ModelHandlerPyTorch as ModelHandler
from idiaptts.src.DataPlotter import DataPlotter
from idiaptts.src.data_preparation.questions.QuestionLabelGen import QuestionLabelGen
from idiaptts.src.data_preparation.PyTorchLabelGensDataset import PyTorchLabelGensDataset
from idiaptts.src.data_preparation.wcad.AtomLabelGen import AtomLabelGen
from idiaptts.src.data_preparation.world.FlatLF0LabelGen import FlatLF0LabelGen
from idiaptts.src.data_preparation.world.WorldFeatLabelGen import WorldFeatLabelGen
from idiaptts.src.model_trainers.wcad.AtomVUVDistPosModelTrainer import AtomVUVDistPosModelTrainer
from idiaptts.src.neural_networks.pytorch.loss.L1WeightedVUVMSELoss import L1WeightedVUVMSELoss


class AtomNeuralFilterModelTrainer(ModelTrainer):
    """
    Implementation of a ModelTrainer for the generation of intonation curves with an end-to-end system.
    The first part of the architecture runs atom position prediction, and the output layer contains neural filters.
    Output curves have dimension: T x 2 (amp, theta).

    Use question labels as input and extracted lf0 as output.
    """
    logger = logging.getLogger(__name__)

    def __init__(self, wcad_root, dir_audio, dir_atom_labels, dir_lf0_labels, dir_question_labels, id_list,
                 thetas, k,
                 num_questions, dist_window_size=51, hparams=None):
        """Default constructor.

        :param wcad_root:               Path to main directory of wcad.
        :param dir_audio:               Path to directory that contains the .wav files.
        :param dir_atom_labels:         Path to directory that contains the .atoms files.
        :param dir_lf0_labels:          Path to directory that contains the .lf0 files.
        :param dir_question_labels:     Path to directory that contains the .lab files.
        :param id_list:                 List containing all ids. Subset is taken as test set.
        :param thetas:                  List of theta values of the used atoms.
        :param k:                       K-value of atoms.
        :param num_questions:           Expected number of questions in question labels.
        :param dist_window_size:        Size of distribution around atom amplitudes when training the atom model.
        :param hparams:                 Hyper-parameter container.
        """

        if hparams is None:
            hparams = self.create_hparams()
            hparams.out_dir = os.path.curdir

        hparams_atom = hparams.hparams_atom

        if hparams_atom is None:
            hparams_atom = copy.deepcopy(hparams)
            hparams_atom.synth_gen_figure = False
            hparams_atom.synth_acoustic_model_path = None
        if hparams.atom_model_path is None:
            hparams.atom_model_path = os.path.join(hparams.out_dir, hparams.networks_dir, hparams.model_name + "_atoms")

        # Write missing default parameters.
        if hparams.variable_sequence_length_train is None:
            hparams.variable_sequence_length_train = hparams.batch_size_train > 1
        if hparams.variable_sequence_length_test is None:
            hparams.variable_sequence_length_test = hparams.batch_size_test > 1
        if hparams.synth_dir is None:
            hparams.synth_dir = os.path.join(hparams.out_dir, "synth")

        super().__init__(id_list, hparams)

        self.InputGen = QuestionLabelGen(dir_question_labels, num_questions)
        self.InputGen.get_normalisation_params(dir_question_labels, hparams.input_norm_params_file_prefix)

        self.OutputGen = FlatLF0LabelGen(dir_lf0_labels, dir_atom_labels)
        self.OutputGen.get_normalisation_params(dir_atom_labels, hparams.output_norm_params_file_prefix)

        self.dataset_train = PyTorchLabelGensDataset(self.id_list_train, self.InputGen, self.OutputGen, hparams, match_lengths=True)
        self.dataset_val = PyTorchLabelGensDataset(self.id_list_val, self.InputGen, self.OutputGen, hparams, match_lengths=True)

        self.atom_trainer = AtomVUVDistPosModelTrainer(wcad_root, dir_atom_labels, dir_lf0_labels,
                                                       dir_question_labels, id_list,
                                                       thetas, k,
                                                       num_questions, dist_window_size,
                                                       hparams_atom)

        if self.loss_function is None:
            self.loss_function = L1WeightedVUVMSELoss(weight_unvoiced=hparams.weight_unvoiced,
                                                      vuv_loss_weight=hparams.vuv_loss_weight,
                                                      L1_loss_weight=hparams.L1_loss_weight,
                                                      reduce=False)
        if hparams.scheduler_type == "default":
            hparams.scheduler_type = "None"

        # Override the collate and decollate methods of batches.
        self.batch_collate_fn = self.prepare_batch
        self.batch_decollate_fn = self.decollate_network_output

    @staticmethod
    def create_hparams(hparams_string=None, verbose=False):
        hparams = ModelTrainer.create_hparams(hparams_string, verbose=False)

        hparams.add_hparams(
            thetas=None,  # One initial theta value per filter.
            k=2,  # Order of the impulse response of the atoms.
            min_atom_amp=0.25,  # Post-processing removes atoms with an absolute amplitude smaller than this.
            complex_poles=True,  # Complex poles possible.
            phase_init=0.0,  # Initial phase of the filters.
            vuv_loss_weight=1.0,  # Weight of the VUV RMSE.
            L1_loss_weight=1.0,  # Weight of the L1 loss on the spiking inputs.
            weight_unvoiced=0.5,  # Weight on unvoiced frames.
            num_questions=None,  # Dimension of the input questions.
            dist_window_size=51,  # Size of distribution around spikes when training the AtomModel.
            atom_model_path=None,  # Path to load a pre-trained atom model from.
            hparams_atom=None  # Hyper-parameter container used in the AtomModelTrainer
        )

        if verbose:
            logging.info(hparams.get_debug_string())

        return hparams

    @staticmethod
    def prepare_batch(batch, common_divisor=1, batch_first=False):
        inputs, targets, seq_lengths_input, seq_lengths_output, mask, permutation = ModelHandler.prepare_batch(batch, common_divisor=common_divisor, batch_first=batch_first)

        if mask is None:
            mask = torch.ones((seq_lengths_output[0], 1, 1))
        mask = mask.expand(*mask.shape[:2], 2)
        # mask: T x B x 2 (lf0, vuv), add L1 error dimension.
        mask = torch.cat((mask, mask[..., -1:]), dim=-1).contiguous()

        # TODO: This is a dirty hack, it works but only for VUV weight of 0 (it completes the loss function WMSELoss).
        mask[..., 0] = mask[..., 0] * seq_lengths_output.float()
        ################################################

        return inputs, targets, seq_lengths_input, seq_lengths_output, mask, permutation

    @staticmethod
    def decollate_network_output(output, _, seq_lengths=None, permutation=None, batch_first=True):
        """Split output into LF0, V/UV and command signals. Return command signals as hidden state."""

        # Split pre-net output (command signals).
        intern_amps, _ = ModelTrainer.split_batch(output[:, :, 2:], None, seq_lengths, permutation, batch_first)
        # Split final LF0, V/UV.
        output, _ = ModelTrainer.split_batch(output[:, :, :2], None, seq_lengths, permutation, batch_first)

        return output, intern_amps

    def init_atom(self, hparams):
        """
        Initialize the atom model.
        If the model_type_filters is None, the old model will be loaded, which already contains the atom model.

        :param hparams:         Hyper-parameter container.
        :return:                Nothing
        """
        if hparams.model_type is None and hparams.hparams_atom.epochs != 0:
                logging.warning("When hparams.model_type=None the old model is loaded. This means that training the "
                                "atom model by hparams.hparams_atom.epochs={} has no effect, so we set it to zero."
                                .format(hparams.hparams_atom.epochs))
                hparams.hparams_atom.epochs = 0

        self.logger.info("Create atom model.")
        self.atom_trainer.init(hparams.hparams_atom)

    def init(self, hparams):
        self.logger.info("Create E2E model.")

        atom_trainer_model_path = os.path.join(hparams.hparams_atom.out_dir,
                                               hparams.hparams_atom.networks_dir,
                                               hparams.hparams_atom.model_name)

        if hparams.hparams_atom.epochs > 0 and hparams.atom_model_path != atom_trainer_model_path:
            logging.warning("Atom model has been trained for {} epochs and saved in {}, "
                            "but you will use hparams.atom_model_path = {} to create a new model."
                            .format(hparams.hparams_atom.epochs, atom_trainer_model_path, hparams.atom_model_path))

        super().init(hparams)

    def train_atom(self, hparams):
        output = self.atom_trainer.train(hparams.hparams_atom)
        if hparams.hparams_atom.epochs > 0:
            self.atom_trainer.benchmark(hparams.hparams_atom)
        return output

    def filters_forward(self, in_tensor, hparams, batch_seq_lengths=None, max_seq_length=None):
        """Get output of each filter without their superposition."""
        self.model_handler.model.eval()

        # If input is numpy array convert it to torch tensor.
        if isinstance(in_tensor, np.ndarray):
            in_tensor = torch.from_numpy(in_tensor)

        if hparams.use_gpu:
            in_tensor = in_tensor.cuda()

        if batch_seq_lengths is None:
            batch_seq_lengths = (len(in_tensor),)

        if max_seq_length is None:
            max_seq_length = max(batch_seq_lengths)

        hidden = self.model_handler.model.init_hidden(len(batch_seq_lengths))
        output = self.model_handler.model.filters_forward(in_tensor, hidden, batch_seq_lengths, max_seq_length)

        return output.detach().cpu().numpy()

    def gen_figure_from_output(self, id_name, output, intern_amps, hparams, clustering=None, filters_out=None):

        if output is None or filters_out is None:
            input_labels = self.InputGen[id_name][:, None, ...]
            output_full = self.model_handler.forward(input_labels, hparams)[0]
            output = output_full[:, 0, :2]
            intern_amps = output_full[:, 0, 2:]
            filters_out = self.filters_forward(input_labels, hparams)[:, 0, ...]

        # Retrieve data from label.
        labels_post = self.OutputGen.postprocess_sample(output)
        output_vuv = labels_post[:, 1]
        output_vuv[output_vuv < 0.5] = 0.0
        output_vuv[output_vuv >= 0.5] = 1.0
        output_vuv = output_vuv.astype(bool)

        output_lf0 = labels_post[:, 0]

        # Load original lf0 and vuv.
        org_labels = self.OutputGen.load_sample(id_name, self.OutputGen.dir_labels)
        original_lf0, _ = self.OutputGen.convert_to_world_features(org_labels)
        # original_lf0, _ = interpolate_lin(original_lf0)

        phrase_curve = self.OutputGen.get_phrase_curve(id_name)
        original_lf0 -= phrase_curve[:len(original_lf0)]
        original_lf0 = original_lf0[:len(output_lf0)]

        org_labels = self.atom_trainer.OutputGen.load_sample(id_name,
                                                             self.atom_trainer.OutputGen.dir_labels,
                                                             len(self.atom_trainer.OutputGen.theta_interval),
                                                             self.atom_trainer.OutputGen.dir_world_labels)
        org_vuv = org_labels[:, 0, 0]
        org_vuv = org_vuv.astype(bool)

        thetas = self.model_handler.model.thetas_approx()

        # Get a data plotter
        net_name = os.path.basename(hparams.model_name)
        filename = str(os.path.join(hparams.out_dir, id_name + '.' + net_name))
        plotter = DataPlotter()

        plot_id = 0

        graphs_intern = list()

        for idx in reversed(range(intern_amps.shape[1])):
            graphs_intern.append((intern_amps[:, idx], r'$\theta$={0:.3f}'.format(thetas[idx])))
        plotter.set_data_list(grid_idx=plot_id, data_list=graphs_intern)
        plotter.set_area_list(grid_idx=plot_id, area_list=[(np.invert(output_vuv), '0.75', 1.0)])
        plotter.set_label(grid_idx=plot_id, ylabel='command')
        amp_max = 0.04
        amp_min = -amp_max
        plotter.set_lim(grid_idx=plot_id, ymin=amp_min, ymax=amp_max)
        plot_id += 1

        graphs_filters = list()
        for idx in reversed(range(filters_out.shape[1])):
            graphs_filters.append((filters_out[:, idx], ))
        plotter.set_data_list(grid_idx=plot_id, data_list=graphs_filters)
        plotter.set_area_list(grid_idx=plot_id, area_list=[(np.invert(output_vuv), '0.75', 1.0, 'Unvoiced')])
        plotter.set_label(grid_idx=plot_id, ylabel='filtered')
        amp_max = 0.1
        amp_min = -amp_max
        plotter.set_lim(grid_idx=plot_id, ymin=amp_min, ymax=amp_max)
        plot_id += 1

        graphs_lf0 = list()
        graphs_lf0.append((original_lf0, "Original"))
        graphs_lf0.append((output_lf0, "Predicted"))
        plotter.set_data_list(grid_idx=plot_id, data_list=graphs_lf0)
        plotter.set_hatchstyles(grid_idx=plot_id, hatchstyles=['\\\\'])
        plotter.set_area_list(grid_idx=plot_id, area_list=[(np.invert(org_vuv.astype(bool)), '0.75', 1.0, 'Reference unvoiced')])
        plotter.set_label(grid_idx=plot_id, xlabel='frames [' + str(hparams.frame_size_ms) + ' ms]',
                          ylabel='lf0')
        amp_lim = 1
        plotter.set_lim(grid_idx=plot_id, ymin=-amp_lim, ymax=amp_lim)
        plotter.set_linestyles(grid_idx=plot_id, linestyles=['-.', '-'])
        plotter.set_colors(grid_idx=plot_id, colors=['C3', 'C2', 'C0'], alpha=1)

        plotter.gen_plot()
        # plotter.gen_plot(True)
        plotter.save_to_file(filename + ".FILTERS" + hparams.gen_figure_ext)

        plotter.plt.show()

        if clustering is None:
            return

        plotter = DataPlotter()

        def cluster(array, mean=False):
            if mean:
                return np.array([np.take(array, i, axis=-1).mean() for i in clustering]).transpose()
            return np.array([np.take(array, i, axis=-1).sum(-1) for i in clustering]).transpose()

        clustered_amps = cluster(intern_amps)
        clustered_thetas = cluster(thetas, True)
        clustered_filters = cluster(filters_out)

        plot_id = 0
        graphs_intern = list()

        for idx in reversed(range(clustered_amps.shape[1])):
            graphs_intern.append((clustered_amps[:, idx], r'$\theta$={0:.3f}'.format(clustered_thetas[idx])))
        plotter.set_data_list(grid_idx=plot_id, data_list=graphs_intern)
        plotter.set_area_list(grid_idx=plot_id, area_list=[(np.invert(output_vuv), '0.75', 1.0, 'Unvoiced')])
        plotter.set_label(grid_idx=plot_id, ylabel='cluster command')
        amp_max = 0.04
        amp_min = -amp_max
        plotter.set_lim(grid_idx=plot_id, ymin=amp_min, ymax=amp_max)
        plot_id += 1

        graphs_filters = list()
        for idx in reversed(range(clustered_filters.shape[1])):
            graphs_filters.append((clustered_filters[:, idx], ))
        plotter.set_data_list(grid_idx=plot_id, data_list=graphs_filters)
        plotter.set_area_list(grid_idx=plot_id, area_list=[(np.invert(output_vuv), '0.75', 1.0)])
        plotter.set_label(grid_idx=plot_id, ylabel='filtered')
        amp_max = 0.175
        amp_min = -amp_max
        plotter.set_lim(grid_idx=plot_id, ymin=amp_min, ymax=amp_max)
        plot_id += 1

        graphs_lf0 = list()
        graphs_lf0.append((original_lf0, "Original"))
        graphs_lf0.append((output_lf0, "Predicted"))
        plotter.set_data_list(grid_idx=plot_id, data_list=graphs_lf0)
        plotter.set_hatchstyles(grid_idx=plot_id, hatchstyles=['\\\\'])
        plotter.set_area_list(grid_idx=plot_id, area_list=[(np.invert(org_vuv.astype(bool)), '0.75', 1.0, 'Reference unvoiced')])
        plotter.set_label(grid_idx=plot_id, xlabel='frames [' + str(hparams.frame_size_ms) + ' ms]',
                          ylabel='lf0')
        # amp_lim = max(np.max(np.abs(wcad_lf0)), np.max(np.abs(output_lf0))) * 1.1
        amp_lim = 1
        plotter.set_lim(grid_idx=plot_id, ymin=-amp_lim, ymax=amp_lim)
        plotter.set_linestyles(grid_idx=plot_id, linestyles=['-.', '-'])
        plotter.set_colors(grid_idx=plot_id, colors=['C3', 'C2', 'C0'], alpha=1)

        plotter.gen_plot()
        # plotter.gen_plot(True)
        plotter.save_to_file(filename + ".CLUSTERS" + hparams.gen_figure_ext)

        plotter.plt.show()

    def gen_figure_atoms(self, hparams, ids_input):
        self.atom_trainer.gen_figure(hparams, ids_input)

    def synthesize(self, id_list, synth_output, hparams):
        """Save output of model to .lf0 and (.vuv) files and call Merlin synth which reads those files."""

        # Reconstruct lf0 from generated atoms and write it to synth output.
        # recon_dict = self.get_recon_from_synth_output(synth_output)
        full_output = dict()
        for id_name, labels in synth_output.items():
            # Take lf0 and vuv from network output.
            lf0 = labels[:, 0]
            vuv = labels[:, 1]

            phrase_curve = self.OutputGen.get_phrase_curve(id_name)
            lf0 = lf0 + phrase_curve[:len(lf0)].squeeze()

            vuv[vuv < 0.5] = 0.0
            vuv[vuv >= 0.5] = 1.0

            # Get mgc, vuv and bap data either through a trained acoustic model or from data extracted from the audio.
            if hparams.synth_acoustic_model_path is None:

                world_dir = hparams.world_dir if hasattr(hparams, "world_dir") and hparams.world_dir is not None \
                                              else os.path.join(self.OutputGen.dir_labels, self.dir_extracted_acoustic_features)
                full_sample: np.ndarray = WorldFeatLabelGen.load_sample(id_name, world_dir,
                                                                        add_deltas=False,
                                                                        num_coded_sps=hparams.num_coded_sps)  # Load extracted data.
                len_diff = len(full_sample) - len(lf0)
                trim_front = len_diff // 2
                trim_end = len_diff - trim_front
                full_sample = WorldFeatLabelGen.trim_end_sample(full_sample, trim_end)
                full_sample = WorldFeatLabelGen.trim_end_sample(full_sample, trim_front, reverse=True)
            else:
                raise NotImplementedError()

            # Overwrite lf0 and vuv by network output.
            full_sample[:, hparams.num_coded_sps] = lf0
            full_sample[:, hparams.num_coded_sps + 1] = vuv
            # Fill a dictionary with the samples.
            full_output[id_name + "_E2E"] = full_sample

        # Run the merlin synthesizer
        self.run_world_synth(full_output, hparams)

    def compute_score(self, dict_outputs_post, dict_hiddens, hparams):

        # Get data for comparision.
        dict_original_post = dict()
        for id_name in dict_outputs_post.keys():
            dict_original_post[id_name] = self.OutputGen.load_sample(id_name, self.OutputGen.dir_labels)

        f0_rmse = 0.0
        vuv_error_rate = 0.0
        f0_rmse_max_id = "None"
        f0_rmse_max = 0.0
        vuv_error_max_id = "None"
        vuv_error_max = 0.0

        all_rmse = []
        all_vuv = []

        for id_name, labels in dict_outputs_post.items():
            output_lf0 = labels[:, 0]
            output_vuv = labels[:, 1]
            output_vuv[output_vuv < 0.5] = 0.0
            output_vuv[output_vuv >= 0.5] = 1.0
            output_vuv = output_vuv.astype(bool)

            # Get data for comparision.
            org_lf0 = dict_original_post[id_name][:, 0]
            org_vuv = dict_original_post[id_name][:, 1]
            phrase_curve = self.OutputGen.get_phrase_curve(id_name)

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
            all_rmse.append(current_f0_rmse)

            num_errors = (org_vuv[:len(output_lf0)] != output_vuv)
            vuv_error_rate_tmp = float(num_errors.sum()) / len(output_lf0)
            if vuv_error_rate_tmp > vuv_error_max:
                vuv_error_max_id = id_name
                vuv_error_max = vuv_error_rate_tmp
            vuv_error_rate += vuv_error_rate_tmp
            all_vuv.append(vuv_error_rate_tmp)

        f0_rmse /= len(dict_outputs_post)
        vuv_error_rate /= len(dict_outputs_post)

        self.logger.info("Worst F0 RMSE: " + f0_rmse_max_id + " {:4.2f}Hz".format(f0_rmse_max))
        self.logger.info("Worst VUV error: " + vuv_error_max_id + " {:2.2f}%".format(vuv_error_max * 100))
        self.logger.info("Benchmark score: F0 RMSE " + "{:4.2f}Hz".format(f0_rmse)
                         + ", VUV " + "{:2.2f}%".format(vuv_error_rate * 100))

        return f0_rmse, vuv_error_rate
