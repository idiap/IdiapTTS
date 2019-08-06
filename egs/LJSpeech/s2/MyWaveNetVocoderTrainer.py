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
import glob
import random
import multiprocessing

# Third-party imports.

# Local source tree imports.
from idiaptts.src.model_trainers.WaveNetVocoderTrainer import WaveNetVocoderTrainer


class MyWaveNetVocoderTrainer(WaveNetVocoderTrainer):
    logger = logging.getLogger(__name__)

    #########################
    # Default constructor
    #
    def __init__(self, hparams):
        """Set your parameters here."""
        dir_world_features = os.path.join(hparams.work_dir, "WORLD")

        # if hparams.voice == "demo":
        #     # DEBUG
        #     with open(os.path.join(os.path.realpath("database"), "file_id_list_test.txt")) as f:
        #         id_list = f.readlines()
        #     id_list[:] = [s.strip(' \t\n\r') for s in id_list]
        #     for index, id_name in enumerate(id_list):
        #         id_list[index] = os.path.join(os.path.realpath("database"), "wav", id_name + ".wav")
        # else:
        # Read which files to process.
        with open(os.path.join(os.path.realpath("database"), "file_id_list_" + hparams.voice + ".txt")) as f:
            id_list = f.readlines()
        id_list[:] = [s.strip(' \t\n\r') for s in id_list]
        # The WaveNet implemenation requires the full path.
        for index, id_name in enumerate(id_list):
            id_list[index] = os.path.join(os.path.realpath(hparams.data_dir), id_name + ".wav")
        # for directory in hparams.data_dir:
        #     id_list += glob.glob(os.path.join(directory, "**", "*.wav"), recursive=True)  # Recursively find all wav files.
        assert(len(id_list) > 0)

        super().__init__(dir_world_features, id_list, hparams)


def main():
    logging.basicConfig(level=logging.INFO)

    hparams = MyWaveNetVocoderTrainer.create_hparams()  # TODO: Parse input for hparams.

    # General parameters
    # hparams.num_questions = 425
    hparams.frame_rate_output_Hz = 16000  # TODO: Same as sampling_frequency? (Synthesised audio is converted to synth_fs.)
    # hparams.sampling_frequency = 16000  # Input audio.
    hparams.frame_size_ms = 5  # ms
    hparams.voice = "full"
    hparams.work_dir = os.path.realpath(os.path.join("experiments", hparams.voice))
    hparams.data_dir = os.path.realpath("database/wav/")
    hparams.out_dir = os.path.join(hparams.work_dir, "r9y9Wavenet")

    # Training parameters.
    hparams.epochs = 40
    hparams.use_gpu = True
    hparams.preload_next_batch_to_gpu = False  # Impossible because of memory requirements.
    hparams.num_gpus = 1
    hparams.batch_size_train = 2 * hparams.num_gpus
    hparams.batch_size_val = 2 * hparams.num_gpus
    hparams.batch_size_test = 2 * hparams.num_gpus
    hparams.batch_size_synth = 1 * hparams.num_gpus
    hparams.test_set_perc = 0.01
    hparams.val_set_perc = 0.01
    hparams.max_input_train_sec = 1.5
    hparams.max_input_test_sec = 2.0
    hparams.use_saved_learning_rate = True  # Don't override learning rate if loaded from checkpoint.
    hparams.learning_rate = 0.001
    hparams.seed = 1234
    hparams.epochs_per_checkpoint = 1
    hparams.use_cond = True
    hparams.start_with_test = False
    hparams.ema_decay = 0.9999

    # Mu-law quantization setup.
    # hparams.input_type = "mulaw-quantize"
    # hparams.quantize_channels = 256
    # hparams.out_channels = 256
    # hparams.mu = 255
    # GMM parameter setup.
    hparams.input_type = "raw"
    hparams.quantize_channels = 65536
    num_mixtures = 4
    hparams.out_channels = num_mixtures * 3  # num_mixtures * 3 (pi, mean, log_scale)

    hparams.layers = 12
    hparams.stacks = 2
    hparams.kernel_size = 2

    # Extra layer for conditional features.
    hparams.upsample_conditional_features = True
    hparams.upsample_scales = [
        1
    ]
    # hparams.layers = 30
    # hparams.stacks = 3
    if hparams.voice == "demo":
        # DEBUG: Small network for tests.
        hparams.layers = 4
        hparams.stacks = 1
        hparams.residual_channels = 8
        hparams.gate_channels = 8
        hparams.skip_out_channels = 8
        hparams.max_input_train_sec = 1.0
        hparams.max_input_test_sec = 1.0

    hparams.model_type = "r9y9WaveNet"
    hparams.model_name = "wn-2-l{}s{}k{}-{}{}{}k-lr_{}{}.nn".format(hparams.layers,
                                                                    hparams.stacks,
                                                                    hparams.kernel_size,
                                                                    ("raw{}-".format(num_mixtures) if not hparams.input_type == "mulaw-quantize" else ""),
                                                                    ("uncond-" if not hparams.use_cond else ""),
                                                                    int(hparams.frame_rate_output_Hz / 1000),
                                                                    hparams.learning_rate,
                                                                    ("-dp" if hparams.num_gpus > 1 else ""))
    # hparams.model_name = "wavenet_vocoder-1-l30-16k-lr_0005-e40.nn"

    trainer = MyWaveNetVocoderTrainer(hparams)
    trainer.init(hparams)
    trainer.train(hparams)

    synth_file_id_list = random.choices(trainer.id_list_test, k=3)
    # path_db = hparams.data_dir
    for index, id_name in enumerate(synth_file_id_list):
        synth_file_id_list[index] = os.path.splitext(id_name)[0]  # Get rid of the .wav in the ids.
        # synth_file_id_list[index] = os.path.join(path_db, id_name[:7], id_name + ".wav")

    trainer.synth(hparams, synth_file_id_list)
    # trainer.synth_ref(synth_file_id_list, hparams)
    # hparams.synth_vocoder = "WORLD"
    # trainer.synth_vocoder(synth_file_id_list, hparams)

    # trainer.benchmark(hparams) Not implemented.


if __name__ == '__main__':
    main()

