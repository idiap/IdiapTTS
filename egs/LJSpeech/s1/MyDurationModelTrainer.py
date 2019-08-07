#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#

"""Module description:
"""

# System imports.
import logging
import sys
import os
import random

# Third-party imports.

# Local source tree imports.
from idiaptts.src.model_trainers.DurationModelTrainer import DurationModelTrainer


class MyDurationModelTrainer(DurationModelTrainer):

    logger = logging.getLogger(__name__)

    #########################
    # Default constructor
    #
    def __init__(self, hparams):
        """Set your parameters here."""

        dir_durations = os.path.join(hparams.work_dir, "dur")
        dir_monophone_labels = os.path.join(hparams.work_dir, "labels", "label_state_align")
        file_symbol_dict = os.path.join(hparams.work_dir, "labels", "mono_phone.list")

        # Read which files to process.
        with open(os.path.join(hparams.data_dir, "file_id_list_" + hparams.voice + ".txt")) as f:
            id_list = f.readlines()
        # Trim entries in-place.
        id_list[:] = [s.strip(' \t\n\r') for s in id_list]

        super().__init__(dir_monophone_labels, dir_durations, id_list, file_symbol_dict, hparams)


def main():
    logging.basicConfig(level=logging.INFO)

    hparams = DurationModelTrainer.create_hparams()  # TODO: Parse input for hparams.

    # General parameters.
    hparams.voice = "full"
    hparams.work_dir = os.path.realpath(os.path.join("experiments", hparams.voice))
    hparams.data_dir = os.path.realpath("database")
    hparams.out_dir = os.path.join(hparams.work_dir, "DurationModel")

    hparams.frame_size_ms = 5
    hparams.seed = 1234

    # Training parameters.
    hparams.epochs = 35
    hparams.use_gpu = True
    hparams.model_type = "RNNDYN-2_TANH_512-2_BiGRU_128-1_FC_5"
    hparams.batch_size_train = 64
    hparams.batch_size_val = 64
    hparams.batch_size_test = 64
    hparams.use_saved_learning_rate = True  # Don't override learning rate if loaded from checkpoint.
    hparams.optimiser_args["lr"] = 0.002
    hparams.grad_clip_norm_type = 2
    hparams.grad_clip_max_norm = 1.0
    hparams.epochs_per_checkpoint = 5
    hparams.scheduler_type = "Plateau"
    hparams.scheduler_args["patience"] = 5
    hparams.ema_decay = 0.99
    hparams.model_name = "DM_b{}_lr{}{}.nn".format(hparams.batch_size_train,
                                                   str(hparams.optimiser_args["lr"]).split('.')[1],
                                                   "_ema" + str(hparams.ema_decay).split('.')[1] if hparams.ema_decay else "")

    trainer = MyDurationModelTrainer(hparams)
    trainer.init(hparams)
    trainer.train(hparams)
    trainer.benchmark(hparams)


if __name__ == "__main__":
    main()
