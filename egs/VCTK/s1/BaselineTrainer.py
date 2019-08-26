#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by 
#

"""Module description:
   Train a baseline multi-speaker system.
   A bi-LSTM network with speaker embedding input in all of its layers.
"""

# System imports.
import logging
import sys
import os

# Third-party imports.

# Local source tree imports.
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd()))))  # Adds the ITTS folder to the path.
from src.model_trainers.AcousticDeltasModelTrainer import AcousticDeltasModelTrainer
from src.neural_networks.pytorch.loss.WMSELoss import WMSELoss
from egs.VCTK.s1 import vctk_utils


class BaselineTrainer(AcousticDeltasModelTrainer):
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
        dir_world_labels = os.path.join(hparams.work_dir, "WORLD")
        dir_question_labels = os.path.join(hparams.work_dir, "questions")

        # # Setup for explicitly splitting by utterances ids.
        # # Read which files to process.
        # with open(os.path.join(hparams.data_dir, "file_id_list_" + hparams.voice + "_train.txt")) as f:
        #     id_list_train = f.readlines()
        # with open(os.path.join(hparams.data_dir, "file_id_list_" + hparams.voice + "_test.txt")) as f:
        #     id_list_test = f.readlines()
        # with open(os.path.join(hparams.data_dir, "file_id_list_" + hparams.voice + "_val.txt")) as f:
        #     id_list_val = f.readlines()
        # # Trim line endings.
        # self.id_list_train = [s.strip(' \t\n\r') for s in id_list_train]
        # self.id_list_test = [s.strip(' \t\n\r') for s in id_list_test]
        # self.id_list_val = [s.strip(' \t\n\r') for s in id_list_val]

        # Setup for pseudo random utterance selection.
        # Read which files to process.
        with open(os.path.join(hparams.data_dir, "file_id_list_" + hparams.voice + "_train.txt")) as f:
            id_list = f.readlines()
        id_list = [s.strip(' \t\n\r') for s in id_list]  # Trim line endings in-place.

        self.loss_function = WMSELoss(hparams.num_coded_sps * 3 + 7, -4, weight=0.0, decision_index_weight=1.0, reduce=False)
        super().__init__(dir_world_labels, dir_question_labels, id_list, hparams.num_questions, hparams)


def main():
    logging.basicConfig(level=logging.INFO)

    hparams = BaselineTrainer.create_hparams()  # TODO: Parse input for hparams.

    # General parameters
    hparams.num_questions = 425
    hparams.voice = "English"
    hparams.work_dir = os.path.realpath(os.path.join("experiments", hparams.voice))
    hparams.data_dir = os.path.realpath("database")
    hparams.out_dir = os.path.join(hparams.work_dir, "BaselineModel")

    hparams.sampling_frequency = 16000
    hparams.frame_size_ms = 5
    hparams.seed = 1234
    hparams.num_coded_sps = 30

    # Training parameters.
    hparams.epochs = 15  # 15
    hparams.use_gpu = True
    hparams.dropout = 0.05
    hparams.batch_size_train = 2  # 32
    hparams.batch_size_val = 48
    hparams.batch_size_benchmark = 48
    hparams.use_saved_learning_rate = True  # Don't override learning rate if loaded from checkpoint.
    hparams.optimiser_args["lr"] = 0.001
    hparams.grad_clip_norm_type = 2
    hparams.grad_clip_max_norm = 100
    hparams.epochs_per_checkpoint = 5
    hparams.start_with_test = True
    hparams.save_final_model = True
    hparams.scheduler_args["patience"] = 5
    hparams.use_best_as_final_model = True

    hparams.model_type = None
    # hparams.model_type = "RNNDYN-33x128_EMB_(-1)-2_RELU_1024-3_BiLSTM_512-1_FC_97"
    hparams.f_get_emb_index = (vctk_utils.id_name_to_speaker_English,)
    # hparams.model_type = "RNNDYN-2_RELU_1024-3_BiLSTM_512-1_FC_97"  # Average model.
    # hparams.f_get_emb_index = None  # No embedding input for average model.

    hparams.model_name = "Bds-{}{}{}-lr{}.nn".format("b{}-".format(hparams.batch_size_train) if hparams.batch_size_train != 32 else "",
                                                     "emb_all" if hparams.f_get_emb_index is not None else "avg",
                                                     "-dropout" + str(hparams.dropout).split('.')[1] if hparams.dropout != 0 else "",
                                                     str(hparams.optimiser_args["lr"]).split('.')[1])

    trainer = BaselineTrainer(hparams)
    trainer.init(hparams)
    trainer.train(hparams)
    trainer.benchmark(hparams)

    # # hparams.synth_gen_figure = False
    # hparams.synth_vocoder = "WORLD"
    # hparams.synth_load_org_sp = False
    # hparams.synth_load_org_bap = False

    # synth_list = dict()
    # synth_list["train"] = ["p225/p225_010", "p226/p226_010", "p239/p239_010"]
    # synth_list["val"] = ["p225/p225_051", "p226/p226_009", "p239/p239_066"]
    # synth_list["test"] = ["p225/p225_033", "p226/p226_175", "p239/p239_056"]
    #
    # # with open(os.path.join(hparams.data_dir, "file_id_list_English_listening_test.txt" + sys.argv[1])) as f:
    # #     id_list_val = f.readlines()
    # # synth_list["val"] = [s.strip(' \t\n\r') for s in id_list_val]  # Trim entries in-place.
    #
    # for key, value in synth_list.items():
    #     hparams.synth_file_suffix = "_" + str(key) + "_" + hparams.synth_vocoder
    #     trainer.synth(hparams, synth_list[key])
    #     trainer.synth_ref(hparams, synth_list[key])
    #     # trainer.gen_figure(hparams, synth_list[key])


if __name__ == "__main__":
    main()
