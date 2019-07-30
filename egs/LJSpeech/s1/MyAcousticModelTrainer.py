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
import os
import random

# Third-party imports.

# Local source tree imports.
from idiaptts.src.model_trainers.AcousticModelTrainer import AcousticModelTrainer
from idiaptts.src.neural_networks.pytorch.loss.WMSELoss import WMSELoss


class MyAcousticModelTrainer(AcousticModelTrainer):
    """
    Implementation of an AcousticModelTrainer with predefined parameters.
    The model predicts MGC, LF0, and BAP with deltas and double deltas and a V/UV flag.
    It uses MLPG to get the final features.

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
        with open(os.path.join(hparams.data_dir, "file_id_list_" + hparams.voice + ".txt")) as f:
            id_list = f.readlines()
        # Trim entries in-place.
        id_list[:] = [s.strip(' \t\n\r') for s in id_list]

        # Use the target V/UV flag to weight the loss of unvoiced frames so that the model focuses on spoken parts.
        self.loss_function = WMSELoss(hparams.num_coded_sps * 3 + 7, -4, weight=0.2, decision_index_weight=1.0, reduce=False)
        super().__init__(dir_world_features, dir_question_labels, id_list, hparams.num_questions, hparams)


def main():
    logging.basicConfig(level=logging.INFO)

    hparams = MyAcousticModelTrainer.create_hparams()  # TODO: Parse input for hparams.

    # General parameters
    hparams.num_questions = 425
    hparams.voice = "full"
    hparams.work_dir = os.path.realpath(os.path.join("experiments", hparams.voice))
    hparams.data_dir = os.path.realpath("database")
    hparams.out_dir = os.path.join(hparams.work_dir, "AcousticModel")

    hparams.num_coded_sps = 30
    hparams.sampling_frequency = 16000
    hparams.frame_size_ms = 5
    hparams.seed = 1234

    # Training parameters.
    hparams.epochs = 15
    hparams.use_gpu = True
    hparams.model_type = "RNNDYN-2_RELU_1024-3_BiLSTM_512-1_FC_97"
    hparams.batch_size_train = 32
    hparams.batch_size_val = 64
    hparams.batch_size_test = 64
    hparams.use_saved_learning_rate = True  # Don't override learning rate if loaded from checkpoint.
    hparams.optimiser_args["lr"] = 0.002
    hparams.grad_clip_norm_type = 2
    hparams.grad_clip_max_norm = 1.0
    hparams.model_name = "AM_b{}_lr{}.nn".format(hparams.batch_size_train, str(hparams.optimiser_args["lr"]).split('.')[1])
    hparams.scheduler_args["patience"] = 5
    hparams.use_best_as_final_model = True

    # Training.
    trainer = MyAcousticModelTrainer(hparams)
    trainer.init(hparams)
    trainer.train(hparams)
    trainer.benchmark(hparams)

    synth_file_id_list = random.choices(trainer.id_list_test, k=3)
    # hparams.synth_gen_figure = False
    hparams.synth_vocoder = "WORLD"
    # hparams.synth_vocoder = "r9y9wavenet_quantized_16k_world_feats_English"
    # hparams.synth_vocoder_path = "wv-1-l24x4-16k-lr_0.005.nn"
    # hparams.synth_file_suffix = "_" + hparams.synth_vocoder

    # with open(os.path.join(hparams.data_dir, "file_id_list_full_eval.txt" + sys.argv[1])) as f:
    #     synth_file_id_list = f.readlines()
    #     # Trim entries in-place.
    # synth_file_id_list[:] = [s.strip(' \t\n\r') for s in synth_file_id_list]

    hparams.synth_gen_figure = True
    trainer.synth(hparams, synth_file_id_list)
    # trainer.synth_ref(synth_file_id_list, hparams)
    # trainer.gen_figure(hparams, synth_file_id_list)


if __name__ == "__main__":
    main()
