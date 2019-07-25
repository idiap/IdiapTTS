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

# Third-party imports.

# Local source tree imports.
from idiaptts.src.model_trainers.AcousticDeltasModelTrainer import AcousticDeltasModelTrainer


class MyAcousticDeltasModelTrainer(AcousticDeltasModelTrainer):
    """
    Implementation of an AcousticDeltasModelTrainer with predefined parameters.

    Use question labels as input and WORLD features as output. Synthesize audio from model output.
    """
    logger = logging.getLogger(__name__)

    #########################
    # Default constructor
    #
    def __init__(self, hparams):
        """Set your parameters here."""
        dir_world_features = os.path.join(hparams.work_dir, "WORLD")
        dir_question_labels = os.path.join(hparams.work_dir, "questions")

        # Read which files to process.
        # with open(os.path.join(hparams.data_dir, "file_id_list_test.txt")) as f:
        with open(os.path.join(hparams.data_dir, "wcad_file_id_list_" + hparams.voice + ".txt")) as f:
            id_list = f.readlines()
        # Trim entries in-place.
        id_list[:] = [s.strip(' \t\n\r') for s in id_list]

        super().__init__(dir_world_features, dir_question_labels, id_list, hparams.num_questions, hparams)


def main():
    logging.basicConfig(level=logging.INFO)

    hparams = MyAcousticDeltasModelTrainer.create_hparams()  # TODO: Parse input for hparams.

    # General parameters
    hparams.num_questions = 425
    hparams.voice = "full"
    hparams.work_dir = os.path.realpath(os.path.join("experiments", hparams.voice))
    hparams.data_dir = os.path.realpath("database")
    hparams.out_dir = os.path.join(hparams.work_dir, "BaselineModel")

    hparams.sampling_frequency = 16000
    hparams.frame_size_ms = 5
    hparams.seed = 1

    # Training parameters.
    hparams.epochs = 35
    hparams.use_gpu = True
    hparams.model_type = "Interspeech18baseline"
    # hparams.model_type = "RNNDYN-2_RELU_1024-3_BiGRU_512-1_FC_187"
    hparams.batch_size_train = 16
    hparams.batch_size_val = 64
    hparams.use_saved_learning_rate = True
    hparams.optimiser_args["lr"] = 0.002
    hparams.grad_clip_norm_type = 2
    hparams.grad_clip_max_norm = 1.0
    hparams.model_name = "Baseline_b{}_lr{}.nn".format(hparams.batch_size_train, str(hparams.optimiser_args["lr"]).split('.')[1])
    hparams.epochs_per_checkpoint = 5

    # Training.
    trainer = MyAcousticDeltasModelTrainer(hparams)
    trainer.init(hparams)
    trainer.train(hparams)
    trainer.benchmark(hparams)

    synth_file_id_list = random.choices(trainer.id_list_test, k=3)
    hparams.synth_gen_figure = True
    hparams.synth_load_org_sp = True
    hparams.synth_load_org_bap = True
    trainer.synth(hparams, synth_file_id_list)

    # # with open(os.path.join(hparams.data_dir, "wcad_file_id_list_full_eval.txt" + sys.argv[1])) as f:
    # with open(os.path.join(hparams.data_dir, "wcad_file_id_list_full_eval.txt")) as f:
    #     id_list_eval = f.readlines()
    #     # Trim entries in-place.
    # id_list_eval[:] = [s.strip(' \t\n\r') for s in id_list_eval]
    # id_list_eval = synth_file_id_list

    # hparams.synth_gen_figure = False
    # trainer.synth(hparams, id_list_eval)
    # trainer.synth_ref(hparams, id_list_eval)


if __name__ == "__main__":
    main()
