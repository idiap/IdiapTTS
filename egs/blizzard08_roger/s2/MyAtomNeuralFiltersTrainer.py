#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#

"""Module description:
   Train a model to predict LF0 from question labels.
   The model consists of a pre-net which is at first trained with the AtomLoss to predict atom spikes.
   Then a NeuralFilters layer is stacked on top of the trained pre-net and everything is trained end-to-end
   with a weighted MSE on the target LF0 where the phrase component is removed.
"""

# System imports.
import sys
import copy
import logging
import os
import numpy as np
import random

# Third-party imports.

# Local source tree imports.
from idiaptts.src.model_trainers.wcad.AtomNeuralFilterModelTrainer import AtomNeuralFilterModelTrainer


class MyAtomNeuralFiltersTrainer(AtomNeuralFilterModelTrainer):
    logger = logging.getLogger(__name__)

    #########################
    # Default constructor
    #
    def __init__(self, hparams):
        """Set your parameters here."""

        dir_wcad_root = "../../../tools/wcad"
        dir_audio = os.path.join(hparams.data_dir, "wav")
        dir_lf0_labels = os.path.join(hparams.work_dir, "WORLD")
        dir_atom_features = os.path.join(hparams.work_dir, "wcad-" + "_".join(map("{:.3f}".format, hparams.thetas)))
        dir_question_labels = os.path.join(hparams.work_dir, "questions")

        # Read which files to process.
        with open(os.path.join(hparams.data_dir, "wcad_file_id_list_" + hparams.voice + ".txt")) as f:
            id_list = f.readlines()
        id_list[:] = [s.strip(' \t\n\r') for s in id_list]

        super().__init__(dir_wcad_root, dir_audio, dir_atom_features, dir_lf0_labels, dir_question_labels, id_list,
                         hparams.thetas, hparams.k, hparams.num_questions, hparams.dist_window_size, hparams)


def main():
    logging.basicConfig(level=logging.INFO)

    hparams = MyAtomNeuralFiltersTrainer.create_hparams()

    # General parameters.
    theta_start = 0.03
    theta_stop = 0.164
    theta_step = 0.015
    hparams.thetas = np.arange(theta_start, theta_stop, theta_step)
    hparams.k = 2
    hparams.min_atom_amp = 0.25
    hparams.num_questions = 425
    hparams.voice = "full"
    hparams.work_dir = os.path.realpath(os.path.join("experiments", hparams.voice))
    hparams.data_dir = os.path.realpath("database")
    hparams.out_dir = os.path.join(hparams.work_dir, "AtomNeuralFilters")

    hparams.frame_size_ms = 5  # [ms]
    hparams.seed = 1
    hparams.dist_window_size = 51  # [frames] should be odd.

    # Training parameters.
    hparams.epochs = 50
    hparams.use_gpu = True
    hparams.model_type = "NeuralFilters"  # If None, old model is loaded by model_name (including atoms network).
    # hparams.model_type = None
    hparams.model_name = "neural_filters.nn"
    hparams.batch_size_train = 5
    hparams.batch_size_val = 5
    hparams.learning_rate = 0.0006
    hparams.complex_poles = False
    hparams.start_with_test = False

    hparams.vuv_loss_weight = 0.1
    hparams.L1_loss_weight = 0.1
    hparams.vuv_weight = 0

    hparams_atom = copy.deepcopy(hparams)
    hparams_atom.synth_gen_figure = False
    hparams_atom.model_type = "RNNDYN-3_RELU_128-2_BiGRU_64-2_RELU_128-1_FC_11"  # If None, old model is loaded by model_name + "_atoms"
    # hparams_atom.model_type = None
    hparams_atom.model_name = hparams.model_name + "_atoms"
    hparams_atom.dropout = 0.0
    hparams_atom.learning_rate = 0.0002
    hparams_atom.batch_size_train = 2
    hparams_atom.epochs = 50  # If 0, model is loaded by hparams.model_name + "_atoms"

    hparams.atom_model_path = os.path.join(hparams.out_dir, hparams.networks_dir, hparams_atom.model_name)
    hparams.hparams_atom = hparams_atom

    # Training
    trainer = MyAtomNeuralFiltersTrainer(hparams)
    trainer.init_atom(hparams)
    trainer.train_atom(hparams)
    trainer.init(hparams)
    trainer.train(hparams)
    trainer.benchmark(hparams)

    synth_file_id_list = random.choices(trainer.id_list_test, k=3)
    # trainer.gen_figure_atoms(synth_file_id_list, hparams_atom)
    trainer.gen_figure(hparams, synth_file_id_list)
    trainer.synth(hparams, synth_file_id_list)
    # trainer.synth_ref(synth_file_id_list, hparams)


if __name__ == "__main__":
    main()
