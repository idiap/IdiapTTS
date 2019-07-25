#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#

"""Module description:
   Train a model to predict mgc, lf0 and bap and synthesize audio from it.
"""

# System imports.
import logging
import sys
import os
import random
import numpy as np

# Third-party imports.
import torch

# Local source tree imports.
from idiaptts.src.model_trainers.wcad.AtomVUVDistPosModelTrainer import AtomVUVDistPosModelTrainer


class MyAtomTrainer(AtomVUVDistPosModelTrainer):
    """
    Implementation of an AtomVUVDistPosModelTrainer with predefined parameters.

    Use question labels as input and atom features as output. Synthesize audio from post-processed model output.
    """
    logger = logging.getLogger(__name__)

    #########################
    # Default constructor
    #
    def __init__(self, hparams):
        """Set your parameters here."""
        dir_wcad_root = "../../../tools/wcad"
        # dir_audio = os.path.join(hparams.data_dir, "wav")
        dir_lf0_labels = os.path.join(hparams.work_dir, "WORLD")
        # theta_step = abs(hparams.thetas[1] - hparams.thetas[0])
        # dir_atom_features = os.path.join(hparams.work_dir, "wcad-" + "_".join(
        #     map("{:.3f}".format, (hparams.thetas[0], hparams.thetas[-1] + theta_step, theta_step))))
        dir_atom_features = os.path.join(hparams.work_dir, "wcad-" + "_".join(map("{:.3f}".format, hparams.thetas)))
        dir_question_labels = os.path.join(hparams.work_dir, "questions")

        # Read which files to process.
        # with open(os.path.join(hparams.data_dir, "file_id_list_test.txt")) as f:
        with open(os.path.join(hparams.data_dir, "wcad_file_id_list_" + hparams.voice + ".txt")) as f:
            id_list = f.readlines()
        # Trim entries in-place.
        id_list[:] = [s.strip(' \t\n\r') for s in id_list]

        super().__init__(dir_wcad_root, dir_atom_features, dir_lf0_labels, dir_question_labels,
                         id_list, hparams.thetas, hparams.k, hparams.num_questions, hparams.dist_window_size,
                         hparams)


def main():
    logging.basicConfig(level=logging.INFO)

    hparams = MyAtomTrainer.create_hparams()

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
    hparams.out_dir = os.path.join(hparams.work_dir, "AtomModel")

    hparams.sampling_frequency = 16000
    hparams.synth_acoustic_model = None  # Set an acoustic model here, not tested.
    hparams.frame_size_ms = 5
    hparams.seed = 1
    hparams.dist_window_size = 51  # [frames] should be odd.

    # Training parameters
    hparams.epochs = 55
    hparams.start_with_test = True
    hparams.use_gpu = True
    hparams.model_type = "RNNDYN-3_RELU_128-2_BiGRU_64-2_RELU_128-1_FC_11"
    hparams.dropout = 0.0
    hparams.batch_size_train = 2
    hparams.batch_size_val = 64
    hparams.use_saved_learning_rate = True  # Don't override learning rate if loaded from checkpoint.
    hparams.optimiser_args["lr"] = 0.0002
    hparams.model_name = "atom_model_b{}_lr{}.nn".format(hparams.batch_size_train, str(hparams.optimiser_args["lr"]).split('.')[1])

    # Training
    trainer = MyAtomTrainer(hparams)
    trainer.init(hparams)
    trainer.train(hparams)
    trainer.benchmark(hparams)

    # with open(os.path.join(os.path.realpath("database"), "wcad_file_id_list_benchmark_1phrase.txt")) as f:
    # with open(os.path.join(hparams_phrase.data_dir, "wcad_file_id_list_full_eval.txt" + sys.argv[1])) as f:
    # with open(os.path.join(hparams.data_dir, "wcad_file_id_list_full_eval.txt")) as f:
    #     id_list_eval = f.readlines()
    # # Trim entries in-place.
    # id_list_eval[:] = [s.strip(' \t\n\r') for s in id_list_eval]

    # Synthesising
    file_id_list_synth = random.choices(trainer.id_list_test, k=3)
    hparams.synth_vocoder = "WORLD"
    trainer.synth_ref(hparams, file_id_list_synth)
    hparams.synth_gen_figure = True
    trainer.synth(hparams, file_id_list_synth)

    # hparams.synth_gen_figure = False
    # trainer.synth(hparams, id_list_eval)
    # trainer.gen_figure(hparams, file_id_list_synth)

    # Benchmarking


if __name__ == "__main__":
    main()
