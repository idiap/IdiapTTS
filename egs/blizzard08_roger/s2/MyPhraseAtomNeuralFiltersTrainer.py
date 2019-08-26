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
   with a weighted MSE on the target LF0 where the phrase component is removed. In the last step the bias
   is added, initialized with the mean LF0 of the speaker computed from the data, and trained end-to-end on
   the target LF0.
"""

# System imports.
import sys
import copy
import logging
import numpy as np
import os
import random

# Local source tree imports.
from idiaptts.src.model_trainers.wcad.PhraseAtomNeuralFilterModelTrainer import PhraseAtomNeuralFilterModelTrainer


class MyPhraseAtomNeuralFiltersTrainer(PhraseAtomNeuralFilterModelTrainer):
    logger = logging.getLogger(__name__)

    #########################
    # Default constructor
    #
    def __init__(self, hparams_phrase):
        """Set your parameters here."""

        hparams_flat = hparams_phrase.hparams_flat

        dir_wcad_root = "../../../tools/wcad"
        dir_audio = os.path.join(hparams_phrase.data_dir, "wav")
        dir_lf0_labels = os.path.join(hparams_phrase.work_dir, "WORLD")
        dir_atom_features = os.path.join(hparams_phrase.work_dir,
                                         "wcad-" + "_".join(map("{:.3f}".format, hparams_flat.thetas)))
        dir_question_labels = os.path.join(hparams_phrase.work_dir, "questions")

        # Read which files to process.
        with open(os.path.join(hparams_phrase.data_dir, "wcad_file_id_list_" + hparams_phrase.voice + ".txt")) as f:
            id_list = f.readlines()
        # Trim entries in-place.
        id_list[:] = [s.strip(' \t\n\r') for s in id_list]

        super().__init__(dir_wcad_root, dir_audio, dir_atom_features, dir_lf0_labels, dir_question_labels, id_list,
                         hparams_flat.thetas, hparams_flat.k, hparams_flat.num_questions, hparams_flat.dist_window_size,
                         hparams_phrase)


def main():
    logging.basicConfig(level=logging.INFO)

    hparams_phrase = MyPhraseAtomNeuralFiltersTrainer.create_hparams()

    hparams_phrase.phrase_bias_init = 4.5 / 1.65  # Mean/std for normalization -> will result in a 4.5 offset in lf0
    hparams_phrase.voice = "full"
    hparams_phrase.work_dir = os.path.realpath(os.path.join("experiments", hparams_phrase.voice))
    hparams_phrase.data_dir = os.path.realpath("database")
    hparams_phrase.out_dir = os.path.join(hparams_phrase.work_dir, "PhraseAtomNeuralFilters")

    hparams_phrase.frame_size_ms = 5  # [ms]
    hparams_phrase.seed = 1
    hparams_phrase.dist_window_size = 51  # [frames] should be odd.

    # Training parameters.
    hparams_phrase.epochs = 50
    hparams_phrase.use_gpu = True
    hparams_phrase.model_type = "PhraseNeuralFilters"  # If None, old model is loaded by model_name (including atoms network).
    # hparams_phrase.model_type = None
    hparams_phrase.model_name = "neural_filters_phrase.nn"
    hparams_phrase.batch_size_train = 5
    hparams_phrase.batch_size_val = 50
    hparams_phrase.optimiser_args["lr"] = 0.0006

    hparams_phrase.vuv_loss_weight = 0.3
    hparams_phrase.L1_loss_weight = 0.3
    hparams_phrase.weight_unvoiced = 0

    hparams_flat = copy.deepcopy(hparams_phrase)

    # General parameters.
    theta_start = 0.03
    theta_stop = 0.164
    theta_step = 0.015
    hparams_flat.thetas = np.arange(theta_start, theta_stop, theta_step)
    hparams_flat.k = 2
    hparams_flat.min_atom_amp = 0.25
    hparams_flat.num_questions = 425

    # Training parameters.
    hparams_flat.epochs = 50
    hparams_flat.model_type = "NeuralFilters"  # If None, old model is loaded by model_name (including atoms network).
    # hparams_flat.model_type = None
    hparams_flat.model_name = hparams_phrase.model_name + "_flat"

    hparams_flat.batch_size_train = 5
    hparams_flat.optimiser_args["lr"] = 0.001
    hparams_flat.complex_poles = True
    hparams_flat.start_with_test = True
    hparams_flat.scheduler_type = "Plateau"

    hparams_atom = copy.deepcopy(hparams_flat)
    hparams_atom.synth_gen_figure = False
    hparams_atom.model_type = "RNNDYN-3_RELU_128-2_BiGRU_64-2_RELU_128-1_FC_11"  # If None, old model is loaded by model_name + "_atoms"
    # hparams_atom.model_type = None
    hparams_atom.model_name = hparams_flat.model_name + "_atoms"
    hparams_atom.dropout = 0.0
    hparams_atom.optimiser_args["lr"] = 0.0002
    hparams_atom.batch_size_train = 2
    hparams_atom.epochs = 50  # If 0, model is loaded by hparams.model_name + "_atoms"
    hparams_atom.train_hidden_init = False

    hparams_phrase.hparams_atom = hparams_atom
    hparams_flat.hparams_atom = hparams_atom
    hparams_phrase.hparams_flat = hparams_flat

    hparams_flat.atom_model_path = os.path.join(hparams_flat.out_dir, hparams_flat.networks_dir,
                                                hparams_atom.model_name)
    hparams_phrase.flat_model_path = os.path.join(hparams_phrase.out_dir, hparams_phrase.networks_dir,
                                                  hparams_flat.model_name)
    hparams_phrase.atom_model_path = hparams_flat.atom_model_path

    # Training
    trainer = MyPhraseAtomNeuralFiltersTrainer(hparams_phrase)
    trainer.init_atom(hparams_phrase)
    trainer.train_atom(hparams_phrase)
    trainer.init_flat(hparams_phrase)
    trainer.train_flat(hparams_phrase)
    trainer.init(hparams_phrase)
    trainer.train(hparams_phrase)
    trainer.benchmark(hparams_phrase)

    synth_file_id_list = random.choices(trainer.id_list_test, k=3)
    trainer.gen_figure_phrase(hparams_flat, synth_file_id_list)
    # trainer.gen_figure_flat(hparams_phrase, synth_file_id_list)
    # trainer.synth(hparams_flat, synth_file_id_list)

    # # Use the following to create samples for the subjective listening test.
    # # First create the wcad_file_id_list_full_test.txt file containing all the test ids randomly selected by the trainer.
    # # with open(os.path.join(hparams_phrase.data_dir, "wcad_file_id_list_full_test.txt" + sys.argv[1])) as f:
    # with open(os.path.join(hparams_phrase.data_dir, "wcad_file_id_list_full_test.txt")) as f:
    #     id_list_eval = f.readlines()
    # id_list_eval[:] = [s.strip(' \t\n\r') for s in id_list_eval]  # Trim line endings in-place.

    id_list_eval = synth_file_id_list

    hparams_phrase.synth_gen_figure = False
    hparams_phrase.synth_vocoder = "WORLD"
    trainer.gen_figure(hparams_phrase, synth_file_id_list)  # Included in synth.
    hparams_phrase.synth_gen_figure = True
    trainer.synth(hparams_phrase, id_list_eval)
    # trainer.synth_ref(hparams_phrase, id_list_eval)

    # Anchor generation by synthesising only the phrase curve.
    hparams_atom.synth_gen_figure = False
    hparams_atom.synth_vocoder = "WORLD"
    # hparams_atom.synth_vocoder = "r9y9wavenet_quantized_16k_world_feats"
    # hparams_atom.synth_file_suffix = "_" + hparams_atom.synth_vocoder
    trainer.flat_trainer.atom_trainer.synth_phrase(id_list_eval, hparams_atom)


if __name__ == "__main__":
    main()
