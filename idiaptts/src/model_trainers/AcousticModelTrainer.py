#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#

"""Module description:
   Train a model to predict mgc, lf0 and bap with deltas and double deltas and synthesize audio from it
   by using MLPG for smoothing.
"""

# System imports.
import logging
import math
import numpy as np
import os

# Third-party imports.
import torch


# Local source tree imports.
from idiaptts.src.model_trainers.ModelTrainer import ModelTrainer
from idiaptts.src.DataPlotter import DataPlotter
from idiaptts.src.Metrics import Metrics
from idiaptts.src.data_preparation.questions.QuestionLabelGen import QuestionLabelGen
from idiaptts.src.data_preparation.world.WorldFeatLabelGen import WorldFeatLabelGen
from idiaptts.src.data_preparation.world.WorldFeatLabelGen import interpolate_lin
from idiaptts.src.data_preparation.PyTorchLabelGensDataset import PyTorchLabelGensDataset as LabelGensDataset


class AcousticModelTrainer(ModelTrainer):
    """
    Implementation of a ModelTrainer for the generation of acoustic data.

    Use question labels as input and WORLD features w/o deltas/double deltas (specified in hparams.add_deltas) as output.
    Synthesize audio from model output with MLPG smoothing.
    """
    logger = logging.getLogger(__name__)

    def __init__(self, dir_world_features, dir_question_labels, id_list, num_questions, hparams=None):
        """Default constructor.

        :param dir_world_features:      Path to the directory containing the world features.
        :param dir_question_labels:     Path to the directory containing the question labels.
        :param id_list:                 List of ids, can contain a speaker directory.
        :param num_questions:           Number of questions in question file.
        :param hparams:                 Set of hyper parameters.
        """
        if hparams is None:
            hparams = self.create_hparams()
            hparams.out_dir = os.path.curdir

        # Write missing default parameters.
        if hparams.variable_sequence_length_train is None:
            hparams.variable_sequence_length_train = hparams.batch_size_train > 1
        if hparams.variable_sequence_length_test is None:
            hparams.variable_sequence_length_test = hparams.batch_size_test > 1
        if hparams.synth_dir is None:
            hparams.synth_dir = os.path.join(hparams.out_dir, "synth")

        super(AcousticModelTrainer, self).__init__(id_list, hparams)

        self.InputGen = QuestionLabelGen(dir_question_labels, num_questions)
        self.InputGen.get_normalisation_params(dir_question_labels, hparams.input_norm_params_file_prefix)

        self.OutputGen = WorldFeatLabelGen(dir_world_features,
                                           add_deltas=hparams.add_deltas,
                                           num_coded_sps=hparams.num_coded_sps,
                                           num_bap=hparams.num_bap,
                                           sp_type=hparams.sp_type)
        self.OutputGen.get_normalisation_params(dir_world_features, hparams.output_norm_params_file_prefix)

        self.dataset_train = LabelGensDataset(self.id_list_train,
                                              self.InputGen,
                                              self.OutputGen,
                                              hparams,
                                              match_lengths=True)
        self.dataset_val = LabelGensDataset(self.id_list_val,
                                            self.InputGen,
                                            self.OutputGen,
                                            hparams,
                                            match_lengths=True)

        if self.loss_function is None:
            self.loss_function = torch.nn.MSELoss(reduction='none')

        if hparams.scheduler_type == "default":
            hparams.scheduler_type = "Plateau"
            hparams.add_hparams(plateau_verbose=True)

    @staticmethod
    def create_hparams(hparams_string=None, verbose=False):
        """Create model hyper parameter container. Parse non default from given string."""
        hparams = ModelTrainer.create_hparams(hparams_string, verbose=False)

        hparams.add_hparams(
            num_questions=None,
            question_file=None,  # Used to add labels in plot.
            num_coded_sps=60,
            sp_type="mcep",
            add_deltas=True,
            synth_load_org_sp=False,
            synth_load_org_lf0=False,
            synth_load_org_vuv=False,
            synth_load_org_bap=False,
            metrics=[Metrics.MCD, Metrics.F0_RMSE, Metrics.VDE, Metrics.BAP_distortion])  # "MCD", "F0 RMSE", "GPE", "FFE", "VDE", "BAP distortion"

        if verbose:
            logging.info(hparams.get_debug_string())

        return hparams

    def gen_figure_from_output(self, id_name, label, hidden, hparams):

        labels_post = self.OutputGen.postprocess_sample(label)
        coded_sp, lf0, vuv, bap = WorldFeatLabelGen.convert_to_world_features(labels_post,
                                                                              contains_deltas=False,
                                                                              num_coded_sps=hparams.num_coded_sps,
                                                                              num_bap=hparams.num_bap)
        lf0, _ = interpolate_lin(lf0)

        # Load original lf0.
        org_labels_post = WorldFeatLabelGen.load_sample(id_name,
                                                        self.OutputGen.dir_labels,
                                                        add_deltas=self.OutputGen.add_deltas,
                                                        num_coded_sps=hparams.num_coded_sps,
                                                        num_bap=hparams.num_bap)
        original_mgc, original_lf0, original_vuv, *_ = WorldFeatLabelGen.convert_to_world_features(
                                                                            org_labels_post,
                                                                            contains_deltas=self.OutputGen.add_deltas,
                                                                            num_coded_sps=hparams.num_coded_sps,
                                                                            num_bap=hparams.num_bap)
        original_lf0, _ = interpolate_lin(original_lf0)

        # Get a data plotter.
        grid_idx = 0
        plotter = DataPlotter()
        net_name = os.path.basename(hparams.model_name)
        filename = str(os.path.join(hparams.out_dir, id_name + '.' + net_name))
        plotter.set_title(id_name + ' - ' + net_name)
        plotter.set_num_colors(3)
        # plotter.set_lim(grid_idx=0, ymin=math.log(60), ymax=math.log(250))
        plotter.set_label(grid_idx=grid_idx, xlabel='frames [' + str(hparams.frame_size_ms) + ' ms]', ylabel='log(f0)')

        graphs = list()
        graphs.append((original_lf0, 'Original lf0'))
        graphs.append((lf0, 'PyTorch lf0'))
        plotter.set_data_list(grid_idx=grid_idx, data_list=graphs)
        plotter.set_area_list(grid_idx=grid_idx,
                              area_list=[(np.invert(vuv.astype(bool)), '0.8', 1.0),
                                         (np.invert(original_vuv.astype(bool)), 'red', 0.2)])

        grid_idx += 1
        import librosa
        plotter.set_label(grid_idx=grid_idx,
                          xlabel='frames [' + str(hparams.frame_size_ms) + ' ms]',
                          ylabel='Original spectrogram')
        plotter.set_specshow(grid_idx=grid_idx,
                             spec=librosa.amplitude_to_db(np.absolute(
                                 WorldFeatLabelGen.mcep_to_amp_sp(original_mgc, hparams.synth_fs)), top_db=None))

        grid_idx += 1
        plotter.set_label(grid_idx=grid_idx,
                          xlabel='frames [' + str(hparams.frame_size_ms) + ' ms]',
                          ylabel='NN spectrogram')
        plotter.set_specshow(grid_idx=grid_idx,
                             spec=librosa.amplitude_to_db(np.absolute(
                                 WorldFeatLabelGen.mcep_to_amp_sp(coded_sp, hparams.synth_fs)), top_db=None))

        if hasattr(hparams, "phoneme_indices") and hparams.phoneme_indices is not None \
           and hasattr(hparams, "question_file") and hparams.question_file is not None:
            questions = QuestionLabelGen.load_sample(id_name,
                                                     "experiments/" + hparams.voice + "/questions/",
                                                     num_questions=hparams.num_questions)[:len(lf0)]
            np_phonemes = QuestionLabelGen.questions_to_phonemes(questions,
                                                                 hparams.phoneme_indices,
                                                                 hparams.question_file)
            plotter.set_annotations(grid_idx, np_phonemes)

        plotter.gen_plot()
        plotter.save_to_file(filename + '.Org-PyTorch' + hparams.gen_figure_ext)

    def compute_score(self, dict_outputs_post, dict_hiddens, hparams):

        dict_original_post = self.get_output_dict(dict_outputs_post.keys(), hparams)

        metric = Metrics(hparams.metrics)
        for id_name, labels in dict_outputs_post.items():
            output = self.OutputGen.convert_to_world_features(sample=labels,
                                                              contains_deltas=False,
                                                              num_coded_sps=hparams.num_coded_sps,
                                                              num_bap=hparams.num_bap)

            org = self.OutputGen.convert_to_world_features(sample=dict_original_post[id_name],
                                                           contains_deltas=self.OutputGen.add_deltas,
                                                           num_coded_sps=hparams.num_coded_sps,
                                                           num_bap=hparams.num_bap)

            current_metrics = metric.get_metrics(hparams.metrics, *org, *output)
            metric.accumulate(id_name, current_metrics)

        metric.log()

        return metric.get_cum_values()

    def get_output_dict(self, id_list, hparams):
        dict_original_post = dict()
        for id_name in id_list:
            dict_original_post[id_name] = WorldFeatLabelGen.load_sample(id_name,
                                                                        dir_out=self.OutputGen.dir_labels,
                                                                        add_deltas=True,
                                                                        num_coded_sps=hparams.num_coded_sps,
                                                                        num_bap=hparams.num_bap)
        return dict_original_post

    def synthesize(self, id_list, synth_output, hparams):
        """
        Depending on hparams override the network output with the extracted features,
        then continue with normal synthesis pipeline.
        """

        if hparams.synth_load_org_sp\
                or hparams.synth_load_org_lf0\
                or hparams.synth_load_org_vuv\
                or hparams.synth_load_org_bap:
            for id_name in id_list:

                world_dir = hparams.world_dir if hasattr(hparams, "world_dir") and hparams.world_dir is not None\
                                              else os.path.join(self.OutputGen.dir_labels,
                                                                self.dir_extracted_acoustic_features)
                labels = WorldFeatLabelGen.load_sample(id_name, world_dir, num_coded_sps=hparams.num_coded_sps,
                                                       num_bap=hparams.num_bap)
                len_diff = len(labels) - len(synth_output[id_name])
                if len_diff > 0:
                    labels = WorldFeatLabelGen.trim_end_sample(labels, int(len_diff / 2), reverse=True)
                    labels = WorldFeatLabelGen.trim_end_sample(labels, len_diff - int(len_diff / 2))

                if hparams.synth_load_org_sp:
                    synth_output[id_name][:len(labels), :self.OutputGen.num_coded_sps] = labels[:, :self.OutputGen.num_coded_sps]

                if hparams.synth_load_org_lf0:
                    synth_output[id_name][:len(labels), -3] = labels[:, -3]

                if hparams.synth_load_org_vuv:
                    synth_output[id_name][:len(labels), -2] = labels[:, -2]

                if hparams.synth_load_org_bap:
                    if hparams.num_bap == 1:
                        synth_output[id_name][:len(labels), -1] = labels[:, -1]
                    else:
                        synth_output[id_name][:len(labels), -hparams.num_bap:] = labels[:, -hparams.num_bap:]

        # Run the vocoder.
        ModelTrainer.synthesize(self, id_list, synth_output, hparams)
