#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by 
#

"""Module description:
   Train a VTLN multi-speaker system from scratch.
   A bi-LSTM network with speaker embedding input in all of its layers plus a VTLN warping layer for
   spectral feature warping is used.
"""


# System imports.
import sys
import logging
import os

# Third-party imports.

# Local source tree imports.
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd()))))  # Adds the ITTS folder to the path.
from src.model_trainers.vtln.VTLNSpeakerAdaptionModelTrainer import VTLNSpeakerAdaptionModelTrainer
from egs.VCTK.s1 import vctk_utils


class VTLNScratchTrainer(VTLNSpeakerAdaptionModelTrainer):
    logger = logging.getLogger(__name__)

    #########################
    # Default constructor
    #
    def __init__(self, hparams):
        """Set your parameters here."""
        dir_world_labels = os.path.join(hparams.work_dir, "WORLD")
        dir_question_labels = os.path.join(hparams.work_dir, "questions")

        # Read which files to process.
        with open(os.path.join(hparams.data_dir, "file_id_list_" + hparams.voice + "_train.txt")) as f:
            id_list = f.readlines()
        id_list[:] = [s.strip(' \t\n\r') for s in id_list]  # Trim line endings in-place.

        # TODO: Ignore unvoiced frames?
        # if hparams.add_deltas:
        #     self.loss_function = WMSELoss(hparams.num_coded_sps * 3 + 7, -4, weight=0.0, decision_index_weight=1.0, reduce=False)
        # else:
        #     self.loss_function = WMSELoss(hparams.num_coded_sps + 3, -2, weight=0.0, decision_index_weight=1.0, reduce=False)
        super().__init__(dir_world_labels, dir_question_labels, id_list, hparams.num_questions, hparams)


def main():
    logging.basicConfig(level=logging.INFO)

    hparams = VTLNScratchTrainer.create_hparams()  # TODO: Parse input for hparams.

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
    hparams.epochs = 0
    hparams.use_gpu = True
    hparams.dropout = 0.05
    hparams.batch_size_train = 2
    hparams.batch_size_val = hparams.batch_size_train
    hparams.batch_size_benchmark = hparams.batch_size_train
    hparams.grad_clip_norm_type = 2
    hparams.grad_clip_max_norm = 100
    hparams.use_saved_learning_rate = True
    hparams.optimiser_args["lr"] = 0.001
    hparams.optimiser_type = "Adam"
    # hparams.optimiser_type = "SGD"
    # hparams.optimiser_args['momentum'] = 0.9
    # hparams.optimiser_args['weight_decay'] = 10E-5
    hparams.scheduler_type = "Plateau"
    hparams.scheduler_args["patience"] = 5
    hparams.start_with_test = True
    hparams.epochs_per_checkpoint = 5
    hparams.save_final_model = True
    hparams.use_best_as_final_model = True

    # hparams.model_type = None
    hparams.model_type = "VTLN"
    hparams.model_name = "VTLN-scratch.nn"
    hparams.pre_net_model_type = "RNNDYN-33x128_EMB_(-1)-2_RELU_1024-3_BiLSTM_512-1_FC_97"
    hparams.pass_embs_to_pre_net = True
    hparams.f_get_emb_index = (vctk_utils.id_name_to_speaker_English,)
    # hparams.pre_net_model_type = "RNNDYN-2_RELU_1024-3_BiLSTM_1024-1_FC_97"  # Pre-net without speaker information.

    # Training.
    trainer = VTLNScratchTrainer(hparams)
    trainer.init(hparams)
    trainer.train(hparams)
    trainer.benchmark(hparams)

    # # hparams.synth_gen_figure = False
    # hparams.synth_vocoder = "WORLD"
    #
    # synth_list = dict()
    # synth_list["train"] = ["p225/p225_010", "p226/p226_010", "p239/p239_010"]
    # synth_list["val"] = ["p225/p225_051", "p226/p226_009", "p239/p239_066"]
    # synth_list["test"] = ["p225/p225_033", "p226/p226_175", "p239/p239_056"]
    #
    # for key, value in synth_list.items():
    #     hparams.synth_file_suffix = "_" + str(key) + "_" + hparams.synth_vocoder
    #     trainer.synth(hparams, synth_list[key])
    #     # trainer.synth_ref(synth_list[key], hparams)
    #     # trainer.gen_figure(synth_list[key], hparams)


if __name__ == "__main__":
    main()
