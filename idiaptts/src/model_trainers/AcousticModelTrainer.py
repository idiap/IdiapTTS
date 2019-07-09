#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>,
#

"""Module description:
   Train a model to predict mgc, lf0 and bap and synthesize audio from it.
"""

# System imports.
import logging
import sys
import numpy as np
import os

# Third-party imports.
import torch

# Local source tree imports.
from idiaptts.src.model_trainers.ModelTrainer import ModelTrainer
from idiaptts.src.DataPlotter import DataPlotter
from idiaptts.src.data_preparation.questions.QuestionLabelGen import QuestionLabelGen
from idiaptts.src.data_preparation.world.WorldFeatLabelGen import WorldFeatLabelGen
from idiaptts.src.data_preparation.world.WorldFeatLabelGen import interpolate_lin
from idiaptts.src.data_preparation.PyTorchLabelGensDataset import PyTorchLabelGensDataset


class AcousticModelTrainer(ModelTrainer):
    """
    Implementation of a ModelTrainer for the generation of acoustic features.

    Use question labels as input and WORLD features as output. Synthesize audio from model output.
    """
    logger = logging.getLogger(__name__)

    #########################
    # Default constructor
    #
    def __init__(self, dir_world_features, dir_question_labels, id_list, num_questions, hparams=None):
        """Default constructor.

        :param dir_world_features:      Path to the directory containing the world features.
        :param dir_question_labels:     Path to the directory containing the question labels.
        :param id_list:                 List containing all ids. Subset is taken as test set.
        :param num_questions:           Expected number of questions in question labels.
        :param hparams:                 Hyper-parameter container.
        """
        if hparams is None:
            hparams = self.create_hparams()
            hparams.out_dir = os.path.curdir

        # Write missing default parameters.
        if hparams.variable_sequence_length_train is None:
            hparams.variable_sequence_length_train = hparams.batch_size_train > 1
        if hparams.variable_sequence_length_test is None:
            hparams.variable_sequence_length_test = hparams.batch_size_test > 1
        if hparams.synth_dir is None:
            hparams.synth_dir = os.path.join(hparams.out_dir, "synth")

        super().__init__(id_list, hparams)

        self.InputGen = QuestionLabelGen(dir_question_labels, num_questions)
        self.InputGen.get_normalisation_params(dir_question_labels)

        self.OutputGen = WorldFeatLabelGen(dir_world_features, add_deltas=False, num_coded_sps=hparams.num_coded_sps)
        self.OutputGen.get_normalisation_params(dir_world_features)

        self.dataset_train = PyTorchLabelGensDataset(self.id_list_train, self.InputGen, self.OutputGen, hparams)
        self.dataset_val = PyTorchLabelGensDataset(self.id_list_val, self.InputGen, self.OutputGen, hparams)

        if self.loss_function is None:
            self.loss_function = torch.nn.MSELoss(reduction='none')

        if hparams.scheduler_type == "default":
            hparams.scheduler_type = "Plateau"
            hparams.plateau_verbose = True

    @staticmethod
    def create_hparams(hparams_string=None, verbose=False):
        hparams = ModelTrainer.create_hparams(hparams_string, verbose=False)
        hparams.num_coded_sps = 60

        if verbose:
            logging.info('Final parsed hparams: %s', hparams.values())

        return hparams

    def gen_figure_from_output(self, id_name, labels, hidden, hparams):

        labels_post = self.OutputGen.postprocess_sample(labels)
        mfcc, lf0, vuv, bap = WorldFeatLabelGen.convert_to_world_features(labels_post, num_coded_sps=self.OutputGen.num_coded_sps)
        lf0, _ = interpolate_lin(lf0)

        # Load original lf0.
        org_labels_post = WorldFeatLabelGen.load_sample(id_name, self.OutputGen.dir_labels, num_coded_sps=hparams.num_coded_sps)
        _, original_lf0, *_ = WorldFeatLabelGen.convert_to_world_features(org_labels_post, num_coded_sps=hparams.num_coded_sps)
        original_lf0, _ = interpolate_lin(original_lf0)

        # Get a data plotter.
        grid_idx = 0
        plotter = DataPlotter()
        net_name = os.path.basename(hparams.model_name)
        filename = str(os.path.join(hparams.out_dir, id_name + '.' + net_name))
        plotter.set_title(id_name + ' - ' + net_name)
        plotter.set_num_colors(3)
        # plotter.set_lim(grid_idx=grid_idx, ymin=math.log(60), ymax=math.log(250))
        plotter.set_label(grid_idx=grid_idx, xlabel='frames [' + str(hparams.frame_size_ms) + ' ms]', ylabel='log(f0)')
        grid_idx += 1

        graphs = list()
        # graphs.append((merlin_lf0, 'Merlin lf0'))
        graphs.append((original_lf0, 'Original lf0'))
        graphs.append((lf0, 'PyTorch lf0'))
        plotter.set_data_list(grid_idx=grid_idx, data_list=graphs)
        plotter.set_area_list(grid_idx=grid_idx, area_list=[(np.invert(vuv.astype(bool)), '0.8', 1.0)])

        if hasattr(hparams, "phoneme_indices") and hparams.phoneme_indices is not None \
           and hasattr(hparams, "question_file") and hparams.question_file is not None:
            questions = QuestionLabelGen.load_sample(id_name,
                                                     "experiments/" + hparams.voice + "/questions/",
                                                     num_questions=hparams.num_questions)[:len(lf0)]
            np_phonemes = QuestionLabelGen.questions_to_phonemes(questions, hparams.phoneme_indices, hparams.question_file)
            plotter.set_annotations(grid_idx, np_phonemes)

        plotter.gen_plot()
        plotter.save_to_file(filename + '.Org-PyTorch' + hparams.gen_figure_ext)


# def main():
#     logging.basicConfig(level=logging.INFO)
#
#     parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#     parser.add_argument("-w", "--dir_world_features",
#                         help="Directory containing the WORLD features in its subdirectories.",
#                         type=str, dest="dir_world_features", default=argparse.SUPPRESS, required=True)
#     parser.add_argument("-l", "--dir_question_labels",
#                         help="Directory containing the question label files.",
#                         type=str, dest="dir_question_labels", default=argparse.SUPPRESS, required=True)
#     parser.add_argument("-q", "--question_file",
#                         help="Full path to question file.",
#                         type=str, dest="question_file", default=argparse.SUPPRESS, required=True)
#     parser.add_argument("-o", "--dir_out",
#                         help="Working directory for loading and saving.",
#                         type=str, dest="dir_out", default=argparse.SUPPRESS, required=True)
#
#     # Synthesis and plotting parameters.
#     parser.add_argument("-s", "--sampling_frequency", help="Sampling frequency of all audio files [Hz].",
#                         type=int, dest="sampling_frequency", choices=[16000, 48000], default=16000)
#     parser.add_argument("-f", "--frame_size_ms", help="Frame size of the labels.",
#                         type=int, dest="frame_size_ms", default=5)
#
#     # Model parameters.
#     parser.add_argument("--model_name",
#                         help="The name used to save the model."
#                              "The model is only saved if training was performed (epochs > 0)."
#                              "If --model_type is not set this name is used to load the model.",
#                         type=str, dest="model_name", default=argparse.SUPPRESS, required=True)
#     parser.add_argument("--model_type",
#                         help="Name of the architecture used for the model."
#                              "If this is not set the model is loaded by its --model_name.",
#                         type=str, dest="model_type", required=False)
#
#     # Training parameters.
#     parser.add_argument("-i", "--file_id_list_path",
#                         help="Path to a text file to read the ids of the files to process.",
#                         type=str, dest="file_id_list_path", default=argparse.SUPPRESS, required=True)
#     parser.add_argument("--use_gpu", help="If set computation is done on a CUDA device.",
#                         dest="use_gpu", action='store_const', const=True, default=False)
#     parser.add_argument("--epochs",
#                         help="Number of epochs in training. If 0 no training is done and model is loaded by name.",
#                         type=int, dest="epochs", default=argparse.SUPPRESS, required=True)
#     parser.add_argument("--batch_size", help="Batch_size is applied to utterances not frames.",
#                         type=int, dest="batch_size", default=1)
#     parser.add_argument("--learning_rate", help="Learning rate to start, but decreases on plateaus.",
#                         type=float, dest="learning_rate", default=0.002)
#
#     # Parse arguments.
#     args = parser.parse_args()
#
#     # Environment/data arguments.
#     dir_world_features = os.path.abspath(args.dir_world_features)
#     dir_question_labels = os.path.abspath(args.dir_question_labels)
#     question_file = os.path.abspath(args.question_file)
#     num_questions = file_len(question_file) + 9
#
#     sampling_frequency = args.sampling_frequency
#     frame_size_ms = args.frame_size_ms
#     if frame_size_ms != parser.get_default("frame_size_ms"):
#         logging.error("Merlin supports only a frame size of 5 ms.")
#         sys.exit(1)
#     file_id_list_path = os.path.abspath(args.file_id_list_path)
#     dir_out = os.path.abspath(args.dir_out)
#     model_type = args.model_type if (hasattr(args, 'model_type') and args.model_type != "None") else None
#     model_name = args.model_name
#
#     # Training arguments.
#     use_gpu = args.use_gpu
#     batch_size = args.batch_size
#     learning_rate = args.learning_rate
#     epochs = max(0, args.epochs)
#
#     # Read which files to process.
#     with open(file_id_list_path) as f:
#         id_list = f.readlines()
#     # Trim entries in-place.
#     id_list[:] = [s.strip(' \t\n\r') for s in id_list]
#
#     acoustic_model_trainer = AcousticModelTrainer(dir_world_features, dir_question_labels, id_list, num_questions, dir_out, sampling_frequency, frame_size_ms)
#     acoustic_model_trainer.train(epochs, use_gpu, model_type, model_name, batch_size, learning_rate)
#
#     synth_file_id_list = ["roger_5535", "roger_5305"]  # , "roger_5604", "roger_6729"]
#     acoustic_model_trainer.synth(False, synth_file_id_list, model_name=model_name)
#
#
# if __name__ == "__main__":
#     main()
