#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#

"""This script synthesises text with a pre-trained duration and acoustic model."""

# System imports.
import os
import argparse
import logging

# Third-party imports.

# Local source tree imports.
import idiaptts
from idiaptts.src.TTSModel import TTSModel
from idiaptts.src.model_trainers.AcousticModelTrainer import AcousticModelTrainer


def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-o", "--out_dir",
                        help="Directory to save the synthesised files to.",
                        type=str, dest="out_dir", default=None, required=False)
    parser.add_argument("-e", "--egs_dir",
                        help="Directory containing this script. Needed to run successfully on the grid.",
                        type=str, dest="egs_dir", default=None, required=False)
    parser.add_argument("input_strings", type=str, nargs='+', help="A text to synthesise.")
    args = parser.parse_args()
    input_strings = args.input_strings
    logging.info("Received {} utterance(s) for synthesis.".format(len(input_strings)))

    hparams = TTSModel.create_hparams()
    hparams.voice = "full"
    hparams.use_gpu = True

    if args.egs_dir is None:
        egs_dir = os.path.dirname(__file__)
        proj_dir = os.path.dirname(os.path.dirname(idiaptts.__file__))
    else:
        egs_dir = os.path.realpath(args.egs_dir)
        proj_dir = os.path.dirname(os.path.dirname(os.path.dirname(egs_dir)))
    tools_dir = os.path.join(proj_dir, "tools")
    tts_frontend_dir = os.path.join(proj_dir, "tools", "tts_frontend")

    hparams.work_dir = os.path.join(egs_dir, "pretrained")
    hparams.synth_dir = os.path.join(egs_dir, "TTSModelDMAM") if args.out_dir is None else os.path.realpath(args.out_dir)

    # Front-end
    hparams.front_end = os.path.join(tts_frontend_dir, "English", "makeLabels.sh")
    hparams.front_end_accent = "AM"
    hparams.festival_dir = os.path.join(tools_dir, "festival")

    # Phoneme
    hparams.file_symbol_dict = os.path.join(hparams.work_dir, "DM_one-hot_ema99-mono_phone.list")
    hparams.min_phoneme_length = 50000
    hparams.num_phoneme_states = 5
    hparams.question_file = os.path.join(tts_frontend_dir, "questions", "questions-en-radio_dnn_416.hed")
    hparams.num_questions = 425
    hparams.question_labels_norm_file = os.path.join(hparams.work_dir, "AM_mgc30-questions-min-max.bin")
    # Duration
    hparams.duration_labels_dir = hparams.work_dir
    hparams.duration_norm_file_name = "DM_one-hot_ema99-dur"
    # Acoustic features
    hparams.world_features_dir = hparams.work_dir
    hparams.num_coded_sps = 30
    # Models
    hparams.duration_model = os.path.join(hparams.work_dir, "DM_one-hot_ema99.nn")
    hparams.acoustic_model = os.path.join(hparams.work_dir, "AM_mgc30.nn")
    # Vocoding
    hparams.synth_vocoder_path = hparams.work_dir
    hparams.synth_vocoder = "WORLD"
    hparams.batch_size_val = 64

    TTSModel.run_DM_AM(hparams, input_strings)


if __name__ == "__main__":
    main()
