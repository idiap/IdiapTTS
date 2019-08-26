#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by 
#

"""Module description:
   Adapt a pre-trained VTLN multi-speaker system to a new speaker by only learning the embedding of the speaker.
   The script sequentially trains two models. One using 380 utterance of the unknown speaker and one using only 10.
   A bi-LSTM network with speaker embedding input in all of its layers plus a VTLN warping layer for
   spectral feature warping is used.
"""


# System imports.
import sys
import logging
import os
import shutil

# Third-party imports.

# Local source tree imports.
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd()))))  # Adds the ITTS folder to the path.
from src.model_trainers.vtln.VTLNSpeakerAdaptionModelTrainer import VTLNSpeakerAdaptionModelTrainer
from egs.VCTK.s1 import vctk_utils


class VTLNAdaptionTrainer(VTLNSpeakerAdaptionModelTrainer):
    logger = logging.getLogger(__name__)

    #########################
    # Default constructor
    #
    def __init__(self, hparams, num_utts_training):
        """Set your parameters here."""
        dir_world_labels = os.path.join(hparams.work_dir, "WORLD")
        dir_question_labels = os.path.join(hparams.work_dir, "questions")

        # Setup for explicitly splitting by utterances ids.
        # Read which files to process.
        with open(os.path.join(hparams.data_dir, "file_id_list_{}_adapt{}_train.txt".format(hparams.voice, num_utts_training))) as f:
            id_list_train = f.readlines()
        with open(os.path.join(hparams.data_dir, "file_id_list_{}_adapt{}_test.txt".format(hparams.voice, num_utts_training))) as f:
            id_list_test = f.readlines()
        with open(os.path.join(hparams.data_dir, "file_id_list_{}_adapt{}_val.txt".format(hparams.voice, num_utts_training))) as f:
            id_list_val = f.readlines()
        # Trim line endings.
        self.id_list_train = [s.strip(' \t\n\r') for s in id_list_train]
        self.id_list_test = [s.strip(' \t\n\r') for s in id_list_test]
        self.id_list_val = [s.strip(' \t\n\r') for s in id_list_val]

        # TODO: Ignore unvoiced frames?
        # if hparams.add_deltas:
        #     self.loss_function = WMSELoss(hparams.num_coded_sps * 3 + 7, -4, weight=0.0, decision_index_weight=1.0, reduce=False)
        # else:
        #     self.loss_function = WMSELoss(hparams.num_coded_sps + 3, -2, weight=0.0, decision_index_weight=1.0, reduce=False)
        super().__init__(dir_world_labels, dir_question_labels, None, hparams.num_questions, hparams)


def main():
    logging.basicConfig(level=logging.INFO)

    hparams = VTLNAdaptionTrainer.create_hparams()  # TODO: Parse input for hparams.

    # General parameters.
    hparams.num_questions = 425
    hparams.voice = "English"
    hparams.work_dir = os.path.realpath(os.path.join("experiments", hparams.voice))
    hparams.data_dir = os.path.realpath("database")
    hparams.out_dir = os.path.join(hparams.work_dir, "VTLNModel")
    hparams.num_speakers = 33
    hparams.speaker_emb_dim = 128
    hparams.sampling_frequency = 16000
    hparams.frame_size_ms = 5
    hparams.seed = 1234
    hparams.num_coded_sps = 30
    hparams.add_deltas = True

    # Training parameters.
    hparams.epochs = 0  # 128
    hparams.use_gpu = True
    hparams.train_pre_net = False
    hparams.pass_embs_to_pre_net = True
    hparams.f_get_emb_index = (vctk_utils.id_name_to_speaker_English,)
    hparams.dropout = 0.05
    hparams.batch_size_train = 2
    hparams.batch_size_val = hparams.batch_size_train
    hparams.batch_size_benchmark = hparams.batch_size_train
    hparams.batch_size_synth = hparams.batch_size_train
    hparams.grad_clip_norm_type = 2
    hparams.grad_clip_max_norm = 100
    hparams.use_saved_learning_rate = False
    hparams.optimiser_args["lr"] = 0.01
    hparams.optimiser_type = "Adam"
    hparams.scheduler_type = "Plateau"
    hparams.scheduler_args["patience"] = 5
    hparams.start_with_test = True
    hparams.epochs_per_checkpoint = 5
    hparams.save_final_model = True
    hparams.use_best_as_final_model = True

    hparams.model_type = None

    # Training.
    source_model_name = "VTLN-emb_all.nn"
    for num_utts_training in [10, 380]:
        hparams.model_name = "VTLN-emb_all-adapt{}.nn".format(num_utts_training)
        nn_dir = os.path.join(hparams.out_dir, "nn")
        if hparams.epochs > 0 or not os.path.isfile(os.path.join(nn_dir, hparams.model_name)):
            logging.info("Copy {} to {}.".format(source_model_name, hparams.model_name))
            shutil.copyfile(os.path.join(nn_dir, source_model_name), os.path.join(nn_dir, hparams.model_name))

        trainer = VTLNAdaptionTrainer(hparams, num_utts_training)
        trainer.init(hparams)
        # Make sure only the embedding is trained.
        for param in trainer.model_handler.model.parameters():
            param.requires_grad = False
        trainer.model_handler.model.embeddings.weight.requires_grad = True
        trainer.train(hparams)
        trainer.benchmark(hparams)
        trainer.benchmark(hparams, "database/file_id_list_English_adapt{}_test_male.txt".format(num_utts_training))
        trainer.benchmark(hparams, "database/file_id_list_English_adapt{}_test_female.txt".format(num_utts_training))

        # # hparams.synth_gen_figure = False
        # hparams.synth_vocoder = "WORLD"
        #
        # # For adaptation to speaker p276.
        # synth_list = dict()
        # synth_list["train"] = ["p276/p276_012"]  # , "p277/p277_012", "p278/p278_012", "p279/p279_012"]
        # synth_list["val"] = ["p276/p276_013"]  # , "p277/p277_013", "p278/p278_013", "p279/p279_013"]
        # synth_list["test"] = ["p276/p276_002", "p276/p276_161",
        #                       "p277/p277_002", "p277/p277_161",
        #                       "p278/p278_002", "p278/p278_161",
        #                       "p279/p279_002", "p279/p279_161"]
        #
        # for key, value in synth_list.items():
        #     hparams.synth_file_suffix = "_" + str(key) + "_" + hparams.synth_vocoder
        #     # trainer.synth(hparams, synth_list[key])
        #     # trainer.synth_ref(hparams, synth_list[key])
        #     trainer.gen_figure(hparams, synth_list[key])

        # # Create synthesised samples for the subjective listening test.
        # if num_utts_training == 10:
        #     with open(os.path.join(hparams.data_dir, "file_id_list_{}_adapt380_listening_test.txt".format(hparams.voice))) as f:
        #         id_list_listening_test = f.readlines()
        #     id_list_listening_test = [s.strip(' \t\n\r') for s in id_list_listening_test]  # Trim line endings.
        #     trainer.synth(hparams, id_list_listening_test)
        #     trainer.synth_ref(hparams, id_list_listening_test)


if __name__ == "__main__":
    main()
