#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#

"""Module description:
   Train a model to predict mgc, lf0 and bap with deltas and double deltas and synthesize audio from it
   by using MLPG for smoothing.
"""

# System imports.
import logging
import math
import sys
import numpy as np
import os

# Third-party imports.
from nnmnkwii import metrics
import torch


# Local source tree imports.
from idiaptts.src.model_trainers.ModelTrainer import ModelTrainer
from idiaptts.src.DataPlotter import DataPlotter
from idiaptts.src.data_preparation.questions.QuestionLabelGen import QuestionLabelGen
from idiaptts.src.data_preparation.world.WorldFeatLabelGen import WorldFeatLabelGen
from idiaptts.src.data_preparation.world.WorldFeatLabelGen import interpolate_lin
from idiaptts.src.data_preparation.PyTorchLabelGensDataset import PyTorchLabelGensDataset as LabelGensDataset


class AcousticDeltasModelTrainer(ModelTrainer):
    """
    Implementation of a ModelTrainer for the generation of acoustic data.

    Use question labels as input and WORLD features with deltas/double deltas as output.
    Synthesize audio from model output with MLPG smoothing.
    """
    logger = logging.getLogger(__name__)

    #########################
    # Default constructor
    #
    def __init__(self, dir_world_features, dir_question_labels, id_list, num_questions, hparams=None):
        """Default constructor.

        :param dir_world_features:      Path to the directory containing the world features.
        :param dir_question_labels:     Path to the directory containing the question labels.
        :param id_list:                 List of ids, can contain a speaker directory.
        :param num_questions:           Number of questions in question file.
        :param hparams:                 Set of hyper parameters.
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

        super(AcousticDeltasModelTrainer, self).__init__(id_list, hparams)

        self.InputGen = QuestionLabelGen(dir_question_labels, num_questions)
        self.InputGen.get_normalisation_params(dir_question_labels)

        self.OutputGen = WorldFeatLabelGen(dir_world_features, add_deltas=True, num_coded_sps=hparams.num_coded_sps)
        self.OutputGen.get_normalisation_params(dir_world_features)

        self.dataset_train = LabelGensDataset(self.id_list_train, self.InputGen, self.OutputGen, hparams)
        self.dataset_val = LabelGensDataset(self.id_list_val, self.InputGen, self.OutputGen, hparams)

        if self.loss_function is None:
            self.loss_function = torch.nn.MSELoss(reduction='none')

        if hparams.scheduler_type == "default":
            hparams.scheduler_type = "Plateau"
            hparams.plateau_verbose = True

    @staticmethod
    def create_hparams(hparams_string=None, verbose=False):
        """Create model hyperparameters. Parse nondefault from given string."""
        hparams = ModelTrainer.create_hparams(hparams_string, verbose=False)

        hparams.synth_load_org_sp = False
        hparams.synth_load_org_lf0 = False
        hparams.synth_load_org_vuv = False
        hparams.synth_load_org_bap = False

        if verbose:
            logging.info('Final parsed hparams: %s', hparams.values())

        return hparams

    # def set_train_params(self, learning_rate):
    #     """Overwrite baseclass method to change non_zero_occurrence value."""
    #
    #     # Defaults for criterion, optimiser_type and scheduler_type.
    #     # If not None it means that the model was loaded from a file.
    #     if self.model_handler.loss_function is not None:
    #         loss_function = self.model_handler.loss_function
    #     else:
    #         loss_function = torch.nn.MSELoss(reduction='none')
    #
    #     if self.model_handler.optimiser is not None:
    #         optimiser = self.model_handler.optimiser
    #         if learning_rate is not None:
    #             for g in optimiser.param_groups:
    #                 g['lr'] = learning_rate
    #     else:
    #         optimiser = torch.optim.Adam(self.model_handler.model.parameters(), lr=learning_rate)
    #
    #     if self.model_handler.scheduler is not None:
    #         scheduler = self.model_handler.scheduler
    #     else:
    #         scheduler = ReduceLROnPlateau(optimiser, verbose=True)
    #
    #     self.model_handler.set_train_params(loss_function, optimiser, scheduler, batch_loss=True)

    def gen_figure_from_output(self, id_name, label, hidden, hparams):

        labels_post = self.OutputGen.postprocess_sample(label)
        coded_sp, lf0, vuv, bap = WorldFeatLabelGen.convert_to_world_features(labels_post, contains_deltas=False, num_coded_sps=hparams.num_coded_sps)
        lf0, _ = interpolate_lin(lf0)

        # Load original lf0.
        org_labels_post = WorldFeatLabelGen.load_sample(id_name, self.OutputGen.dir_labels, add_deltas=self.OutputGen.add_deltas, num_coded_sps=hparams.num_coded_sps)
        original_mgc, original_lf0, original_vuv, *_ = WorldFeatLabelGen.convert_to_world_features(org_labels_post, contains_deltas=self.OutputGen.add_deltas, num_coded_sps=hparams.num_coded_sps)
        original_lf0, _ = interpolate_lin(original_lf0)

        # # Load merlin lf0.
        # merlin_data_dir = os.path.join(os.path.dirname(self.InputGen.label_dir), "wav_merlin/")
        # with open(os.path.join(merlin_data_dir, id_name + ".lf0"), 'rb') as f:
        #     merlin_lf0 = np.fromfile(f, dtype=np.float32)
        #     merlin_lf0 = np.reshape(merlin_lf0, merlin_lf0.shape[0] / original_lf0.shape[1], original_lf0.shape[1])
        #     merlin_lf0[merlin_lf0 < 0] = 0.0
        #     merlin_lf0, _ = interpolate_lin(merlin_lf0)

        # Get a data plotter.
        grid_idx = 0
        plotter = DataPlotter()
        net_name = os.path.basename(hparams.model_name)
        filename = str(os.path.join(hparams.out_dir, id_name + '.' + net_name))
        plotter.set_title(id_name + ' - ' + net_name)
        plotter.set_num_colors(3)
        # plotter.set_lim(grid_idx=0, ymin=math.log(60), ymax=math.log(250))
        plotter.set_label(grid_idx=grid_idx, xlabel='frames [' + str(hparams.frame_size_ms) + ' ms]', ylabel='log(f0)')

        graphs = list()
        # graphs.append((merlin_lf0, 'Merlin lf0'))
        graphs.append((original_lf0, 'Original lf0'))
        graphs.append((lf0, 'PyTorch lf0'))
        plotter.set_data_list(grid_idx=grid_idx, data_list=graphs)
        plotter.set_area_list(grid_idx=grid_idx, area_list=[(np.invert(vuv.astype(bool)), '0.8', 1.0),(np.invert(original_vuv.astype(bool)), 'red', 0.2)])

        grid_idx += 1
        plotter.set_label(grid_idx=grid_idx, xlabel='frames [' + str(hparams.frame_size_ms) + ' ms]', ylabel='Original spectrogram')
        plotter.set_specshow(grid_idx=grid_idx, spec=WorldFeatLabelGen.mgc_to_sp(original_mgc, hparams.synth_fs))

        grid_idx += 1
        plotter.set_label(grid_idx=grid_idx, xlabel='frames [' + str(hparams.frame_size_ms) + ' ms]', ylabel='NN spectrogram')
        plotter.set_specshow(grid_idx=grid_idx, spec=WorldFeatLabelGen.mgc_to_sp(coded_sp, hparams.synth_fs))

        if hasattr(hparams, "phoneme_indices") and hparams.phoneme_indices is not None \
           and hasattr(hparams, "question_file") and hparams.question_file is not None:
            questions = QuestionLabelGen.load_sample(id_name,
                                                     "experiments/" + hparams.voice + "/questions/",
                                                     num_questions=hparams.num_questions)[:len(lf0)]
            np_phonemes = QuestionLabelGen.questions_to_phonemes(questions, hparams.phoneme_indices, hparams.question_file)
            plotter.set_annotations(grid_idx, np_phonemes)

        plotter.gen_plot()
        plotter.save_to_file(filename + '.Org-PyTorch' + hparams.gen_figure_ext)

    def compute_score(self, dict_outputs_post, dict_hiddens, hparams):

        # Get data for comparision.
        dict_original_post = dict()
        for id_name in dict_outputs_post.keys():
            dict_original_post[id_name] = WorldFeatLabelGen.load_sample(id_name, self.OutputGen.dir_labels, True, num_coded_sps=hparams.num_coded_sps)

        f0_rmse = 0.0
        f0_rmse_max_id = "None"
        f0_rmse_max = 0.0
        all_rmse = []
        vuv_error_rate = 0.0
        vuv_error_max_id = "None"
        vuv_error_max = 0.0
        all_vuv = []
        mcd = 0.0
        mcd_max_id = "None"
        mcd_max = 0.0
        all_mcd = []
        bap_error = 0.0
        bap_error_max_id = "None"
        bap_error_max = 0.0
        all_bap_error = []

        for id_name, labels in dict_outputs_post.items():
            output_coded_sp, output_lf0, output_vuv, output_bap = self.OutputGen.convert_to_world_features(labels, False, num_coded_sps=hparams.num_coded_sps)
            output_vuv = output_vuv.astype(bool)

            # Get data for comparision.
            org_coded_sp, org_lf0, org_vuv, org_bap = self.OutputGen.convert_to_world_features(dict_original_post[id_name],
                                                                                               self.OutputGen.add_deltas,
                                                                                               num_coded_sps=hparams.num_coded_sps)

            # Compute f0 from lf0.
            org_f0 = np.exp(org_lf0.squeeze())[:len(output_lf0)]  # Fix minor negligible length mismatch.
            output_f0 = np.exp(output_lf0)

            # Compute MCD.
            org_coded_sp = org_coded_sp[:len(output_coded_sp)]
            current_mcd = metrics.melcd(output_coded_sp, org_coded_sp)  # TODO: Use aligned mcd.
            if current_mcd > mcd_max:
                mcd_max_id = id_name
                mcd_max = current_mcd
            mcd += current_mcd
            all_mcd.append(current_mcd)

            # Compute RMSE.
            f0_mse = (org_f0 - output_f0) ** 2
            current_f0_rmse = math.sqrt((f0_mse * org_vuv[:len(output_lf0)]).sum() / org_vuv[:len(output_lf0)].sum())
            if current_f0_rmse != current_f0_rmse:
                logging.error("Computed NaN for F0 RMSE for {}.".format(id_name))
            else:
                if current_f0_rmse > f0_rmse_max:
                    f0_rmse_max_id = id_name
                    f0_rmse_max = current_f0_rmse
                f0_rmse += current_f0_rmse
                all_rmse.append(current_f0_rmse)

            # Compute error of VUV in percentage.
            num_errors = (org_vuv[:len(output_lf0)] != output_vuv)
            vuv_error_rate_tmp = float(num_errors.sum()) / len(output_lf0)
            if vuv_error_rate_tmp > vuv_error_max:
                vuv_error_max_id = id_name
                vuv_error_max = vuv_error_rate_tmp
            vuv_error_rate += vuv_error_rate_tmp
            all_vuv.append(vuv_error_rate_tmp)

            # Compute aperiodicity distortion.
            org_bap = org_bap[:len(output_bap)]
            if len(output_bap.shape) > 1 and output_bap.shape[1] > 1:
                current_bap_error = metrics.melcd(output_bap, org_bap)  # TODO: Use aligned mcd?
            else:
                current_bap_error = math.sqrt(((org_bap - output_bap) ** 2).mean()) * (10.0 / np.log(10) * np.sqrt(2.0))
            if current_bap_error > bap_error_max:
                bap_error_max_id = id_name
                bap_error_max = current_bap_error
            bap_error += current_bap_error
            all_bap_error.append(current_bap_error)

        f0_rmse /= len(dict_outputs_post)
        vuv_error_rate /= len(dict_outputs_post)
        mcd /= len(dict_original_post)
        bap_error /= len(dict_original_post)

        self.logger.info("Worst MCD: {} {:4.2f}dB".format(mcd_max_id, mcd_max))
        self.logger.info("Worst F0 RMSE: {} {:4.2f}Hz".format(f0_rmse_max_id, f0_rmse_max))
        self.logger.info("Worst VUV error: {} {:2.2f}%".format(vuv_error_max_id, vuv_error_max * 100))
        self.logger.info("Worst BAP error: {} {:4.2f}db".format(bap_error_max_id, bap_error_max))
        self.logger.info("Benchmark score: MCD {:4.2f}dB, F0 RMSE {:4.2f}Hz, VUV {:2.2f}%, BAP error {:4.2f}db".format(mcd, f0_rmse, vuv_error_rate * 100, bap_error))

        return mcd, f0_rmse, vuv_error_rate, bap_error

    def synthesize(self, id_list, synth_output, hparams):
        """Depending on hparams override the network output with the extracted features, then continue with normal synthesis pipeline."""

        if hparams.synth_load_org_sp or hparams.synth_load_org_lf0 or hparams.synth_load_org_vuv or hparams.synth_load_org_bap:
            for id_name in id_list:

                labels = WorldFeatLabelGen.load_sample(id_name, os.path.join(self.OutputGen.dir_labels, self.dir_extracted_acoustic_features), num_coded_sps=hparams.num_coded_sps)
                len_diff = len(labels) - len(synth_output[id_name])
                if len_diff > 0:
                    labels = WorldFeatLabelGen.trim_end_sample(labels, int(len_diff / 2), reverse=True)
                    labels = WorldFeatLabelGen.trim_end_sample(labels, len_diff - int(len_diff / 2))

                if hparams.synth_load_org_sp:
                    synth_output[id_name][:len(labels), :self.OutputGen.num_coded_sps] = labels[:, :self.OutputGen.num_coded_sps]

                if hparams.synth_load_org_lf0:
                    synth_output[id_name][:len(labels), -3] = labels[:, -3]

                if hparams.synth_load_org_vuv:
                    synth_output[id_name][:len(labels), -2] = labels[:, -2]

                if hparams.synth_load_org_bap:
                    synth_output[id_name][:len(labels), -1] = labels[:, -1]

        # Run the vocoder.
        ModelTrainer.synthesize(self, id_list, synth_output, hparams)


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
#     frame_size = args.frame_size
#     if frame_size != parser.get_default("frame_size_ms"):
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
#     acoustic_model_trainer = AcousticDeltasModelTrainer(dir_world_features, dir_question_labels, id_list, num_questions, dir_out, sampling_frequency, frame_size)
#     acoustic_model_trainer.train(epochs, use_gpu, model_type, model_name, batch_size, learning_rate)
#
#     synth_file_id_list = ["roger_5535", "roger_5305"]  # , "roger_5604", "roger_6729"]
#     acoustic_model_trainer.synth(False, synth_file_id_list, model_name=model_name)
#
#
# if __name__ == "__main__":
#     main()
