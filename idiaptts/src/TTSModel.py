#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#


# System imports.
import tempfile
import os
import sys
import subprocess
import shutil
import numpy as np
import logging

# Third-party imports.

# Local source tree imports
from idiaptts.misc.utils import makedirs_safe
from idiaptts.src.model_trainers.DurationModelTrainer import DurationModelTrainer
from idiaptts.src.data_preparation.questions.QuestionLabelGen import QuestionLabelGen
from idiaptts.src.model_trainers.AcousticModelTrainer import AcousticModelTrainer


class TTSModel(object):
    """This class provides static methods to run TTS (text-to-speech) for different setups."""

    @staticmethod
    def create_hparams(hparams_string=None, verbose=False):
        # Combine parameters needed for training acoustic and duration model.
        hparams = AcousticModelTrainer.create_hparams(hparams_string, verbose=False)
        hparams_duration = DurationModelTrainer.create_hparams(hparams_string, verbose=False)
        hparams.override_from_hparam(hparams_duration)

        # Add parameters required for full TTS.
        hparams.add_hparams(
            front_end=None,
            front_end_accent=None,
            festival_dir=None,
            file_symbol_dict=None,
            num_phoneme_states=None,
            duration_labels_dir=None,
            duration_norm_file_name=None,
            duration_model=None,
            question_labels_norm_file=None,
            world_features_dir=None,
            acoustic_model=None,
            synth_load_org_lf0=False,
            synth_load_org_vuv=False,
            synth_load_org_bap=False)

        if verbose:
            logging.info('Final parsed hparams: %s', hparams.values())

        return hparams

    @staticmethod
    def run_DM_AM(hparams, input_strings):
        """
        A function for TTS with a pre-trained duration and acoustic model.

        :param hparams:            Hyper-parameter container. The following parameters are used:
                                   front_end:                    Full path to the makeLabels.sh script in tools/tts_frontend, depends on the language.
                                   festival_dir:                 Full path to the directory with the festival bin/ folder.
                                   front_end_accent (optional):  Give an accent to the front_end, used in tts_frontend.
                                   duration_labels_dir:          Full path to the folder containing the normalisation parameters used to train the duration model.
                                   file_symbol_dict:             A file containing all the used phonemes (has been used to train the duration model, usually mono_phone.list).
                                   duration_model:               Full path to the pre-trained duration model.
                                   num_phoneme_states:           Number of states per phoneme, for each a duration is predicted by the duration model.
                                   question_file:               Full path to question file used to train the acoustic model.
                                   question_labels_norm_file:    Full path to normalisation file of questions used to train the acoustic model.
                                   num_questions:                Number of questions which form the input dimension to the acoustic model.
                                   acoustic_model:               Full path to acoustic model.
        :param input_strings:
        :return:
        """
        # Create a temporary directory to store all files.
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            # tmp_dir_name = os.path.realpath("TMP")
            # makedirs_safe(tmp_dir_name)
            hparams.out_dir = tmp_dir_name
            print("Created temporary directory", tmp_dir_name)
            id_list = ["synth" + str(idx) for idx in range(len(input_strings))]

            # Write the text to synthesise into a single synth.txt file with ids.
            utts_file = os.path.join(tmp_dir_name, "synth.txt")
            with open(utts_file, "w") as text_file:
                for idx, text in enumerate(input_strings):
                    text_file.write("synth{}\t{}\n".format(idx, text))  # TODO: Remove parenthesis etc.

            # Call the front end on the synth.txt file.
            front_end_arguments = [hparams.front_end, hparams.festival_dir, utts_file]
            if hasattr(hparams, "front_end_accent") and hparams.front_end_accent is not None:
                front_end_arguments.append(hparams.front_end_accent)
            front_end_arguments.append(tmp_dir_name)
            subprocess.check_call(front_end_arguments)

            # Remove durations from mono labels.
            dir_mono_no_align = os.path.join(tmp_dir_name, "mono_no_align")
            dir_mono = os.path.join(tmp_dir_name, "labels", "mono")

            if os.path.isdir(dir_mono_no_align):
                shutil.rmtree(dir_mono_no_align)
            os.rename(dir_mono, dir_mono_no_align)
            for id_name in id_list:
                with open(os.path.join(dir_mono_no_align, id_name + ".lab"), "r") as f:
                    old = f.read()
                    monophones = old.split()[2::3]
                with open(os.path.join(dir_mono_no_align, id_name + ".lab"), "w") as f:
                    f.write("\n".join(monophones))

            # Run duration model.
            hparams.batch_size_test = len(input_strings)
            hparams.test_set_perc = 0.0
            hparams.val_set_perc = 0.0
            hparams.phoneme_label_type = "mono_no_align"
            hparams.output_norm_params_file_prefix = hparams.duration_norm_file_name if hasattr(hparams, "duration_norm_file_name") else None
            duration_model_trainer = DurationModelTrainer(os.path.join(tmp_dir_name, "mono_no_align"),
                                                          hparams.duration_labels_dir, id_list,
                                                          hparams.file_symbol_dict, hparams)
            hparams.model_path = hparams.duration_model

            # Predict durations. Durations are already converted to multiples of hparams.min_phoneme_length.
            duration_model_trainer.init(hparams)
            _, output_dict_post = duration_model_trainer.forward(hparams, id_list)
            hparams.output_norm_params_file_prefix = None  # Reset again.

            # Write duration to full labels.
            dir_full = os.path.join(tmp_dir_name, "labels", "full")
            dir_label_state_align = os.path.join(tmp_dir_name, "labels", "label_state_align")
            makedirs_safe(dir_label_state_align)
            for id_name in id_list:
                with open(os.path.join(dir_full, id_name + ".lab"), "r") as f:
                    full = f.read().split()[2::3]
                with open(os.path.join(dir_label_state_align, id_name + ".lab"), "w") as f:
                    current_time = 0
                    timings = output_dict_post[id_name]
                    for idx, monophone in enumerate(full):
                        for state in range(hparams.num_phoneme_states):
                            next_time = current_time + int(timings[idx, state])
                            f.write("{}\t{}\t{}[{}]\n".format(current_time, next_time, monophone, state + 2))
                            current_time = next_time

            # Generate questions from HTK full labels.
            QuestionLabelGen.gen_data(dir_label_state_align, hparams.question_file, dir_out=tmp_dir_name, file_id_list="synth", id_list=id_list, return_dict=False)

            # Run acoustic model and synthesise.
            shutil.copy2(hparams.question_labels_norm_file, tmp_dir_name + "/min-max.bin")  # Get normalisation parameters in same directory.
            acoustic_model_trainer = AcousticModelTrainer(hparams.world_features_dir, tmp_dir_name, id_list, hparams.num_questions, hparams)
            hparams.model_path = hparams.acoustic_model
            acoustic_model_trainer.init(hparams)
            hparams.model_name = ""  # No suffix in synthesised files.
            _, output_dict_post = acoustic_model_trainer.synth(hparams, id_list)

        return 0


# def main():
#     hparams = AcousticModelTrainer.create_hparams()
#
#     hparams.voice = "demo"
#     hparams.work_dir = os.path.realpath(os.path.join("experiments", hparams.voice))
#     hparams.synth_dir = os.path.join(hparams.work_dir, "TTSModel")
#     hparams.use_gpu = True
#
#     hparams.front_end = "../../../tools/tts_frontend/English/makeLabels.sh"
#     hparams.front_end_accent = "AM"
#     hparams.festival_dir = "../../../tools/festival/"
#     hparams.duration_labels_dir = os.path.join(hparams.work_dir, "dur")
#     hparams.duration_model = os.path.join(hparams.work_dir, "DurationModel", "nn", "DM-1-b64.nn")
#     hparams.file_symbol_dict = os.path.join(hparams.work_dir, "labels", "mono_phone.list")
#     hparams.min_phoneme_length = 50000
#     hparams.num_phoneme_states = 5
#     hparams.question_file = "../../../tools/tts_frontend/questions/questions-en-radio_dnn_416.hed"
#     hparams.num_questions = 425
#     hparams.question_labels_norm_file = os.path.join(hparams.work_dir, "questions", "min-max.bin")
#     hparams.world_features_dir = os.path.join(hparams.work_dir, "WORLD")
#     hparams.num_coded_sps = 30
#     hparams.acoustic_model = os.path.join(hparams.work_dir, "AcousticDeltasModel", "nn", "Bds.nn")
#
#     TTSModel.run_DM_AM(hparams, ["This is a test.", "Hello World."])
#
#
# if __name__ == "__main__":
#     main()
