#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>,
# François Marelli <francois.marelli@idiap.ch>
#

"""Module description:
   Train a model to predict atoms.
"""

# System imports.
import logging
import sys
import os
import numpy as np

# Third-party imports.
import torch

# Local source tree imports.
from idiaptts.src.model_trainers.wcad.AtomModelTrainer import AtomModelTrainer


class MyAtomBaseTrainer(AtomModelTrainer):
    """
    Implementation of an AtomModelTrainer with predefined parameters.

    Use question labels as input and atom features as output.
    Synthesize audio from post-processed model output with original spectral and bap features.
    """
    logger = logging.getLogger(__name__)

    #########################
    # Default constructor
    #
    def __init__(self, hparams):
        """Set your parameters here."""
        dir_wcad_root = "../../../tools/wcad"
        # theta_step = abs(hparams.thetas[1]-hparams.thetas[0])
        # dir_atom_features = os.path.join(hparams.work_dir, "wcad-" + "_".join(map("{:.3f}".format, (hparams.thetas[0], hparams.thetas[-1] + theta_step, theta_step))))
        dir_atom_features = os.path.join(hparams.work_dir, "wcad-" + "_".join(map("{:.3f}".format, hparams.thetas)))
        dir_question_labels = os.path.join(hparams.work_dir, "questions")

        # Read which files to process.
        # with open(os.path.join(hparams.data_dir, "file_id_list_test.txt")) as f:
        with open(os.path.join(hparams.data_dir, "wcad_file_id_list_" + hparams.voice + ".txt")) as f:
            id_list = f.readlines()
        # Trim entries in-place.
        id_list[:] = [s.strip(' \t\n\r') for s in id_list]

        super().__init__(dir_wcad_root, dir_atom_features, dir_question_labels,
                         id_list, hparams.thetas, hparams.k, hparams.num_questions,
                         hparams)


def main():
    logging.basicConfig(level=logging.INFO)

    hparams = MyAtomBaseTrainer.create_hparams()

    # General parameters.
    theta_start = 0.01
    theta_stop = 0.055
    theta_step = 0.005
    hparams.thetas = np.arange(theta_start, theta_stop, theta_step)
    hparams.k = 6
    hparams.min_atom_amp = 0.25
    hparams.num_questions = 425
    hparams.voice = "full"
    hparams.work_dir = os.path.realpath(os.path.join("experiments", hparams.voice))
    hparams.data_dir = os.path.realpath("database")
    hparams.out_dir = os.path.join(hparams.work_dir, "AtomBaseModel")

    # Training parameters.
    hparams.epochs = 25
    hparams.use_gpu = True
    hparams.model_type = "RNNDYN-3_RELU_128-2_BiGRU_64-1_FC_9"
    hparams.model_name = "atom_base_model.nn"
    hparams.batch_size_train = 1
    hparams.batch_size_val = 64
    hparams.use_saved_learning_rate = True  # Don't override learning rate if loaded from checkpoint.
    hparams.optimiser_args["lr"] = 0.0002
    hparams.epochs_per_checkpoint = 5

    hparams.sampling_frequency = 16000
    hparams.frame_size_ms = 5
    hparams.seed = 1234

    trainer = MyAtomBaseTrainer(hparams)
    trainer.init(hparams)
    trainer.train(hparams)

    synth_file_id_list = ["roger_5535", "roger_5302"]  # , "roger_5604", "roger_6729"]

    trainer.synth(hparams, synth_file_id_list)
    trainer.gen_figure(hparams, synth_file_id_list)
    trainer.benchmark(hparams)


if __name__ == "__main__":
    main()
