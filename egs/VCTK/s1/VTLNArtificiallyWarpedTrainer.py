#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by 
#

"""Module description:
   First, train a baseline (bi-LSTM) on speaker p276. Second, initialize a VTLN network with that baseline.
   Third, train the VTLN network (with fixed pre-net) on a "new" speaker, which is the same speaker but with its
   spectrogram warped with a random alpha for each phoneme. At last, compare how much of the MCD error is compensated
   by the learned warping. This experiment shows that the VTLN layer can learn a phoneme dependent alpha.
"""

# System imports.
import math
import sys
import logging
import os
import numpy as np

# Third-party imports.
import torch
from nnmnkwii import metrics

# Local source tree imports.
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd()))))  # Adds the ITTS folder to the path.
from src.model_trainers.vtln.VTLNSpeakerAdaptionModelTrainer import VTLNSpeakerAdaptionModelTrainer
from src.model_trainers.AcousticDeltasModelTrainer import AcousticDeltasModelTrainer
from misc.utils import interpolate_lin
from src.neural_networks.pytorch.loss.WMSELoss import WMSELoss
from src.data_preparation.world.WorldFeatLabelGen import WorldFeatLabelGen
from src.data_preparation.questions.QuestionLabelGen import QuestionLabelGen
from src.DataPlotter import DataPlotter
from src.neural_networks.pytorch.models.WarpingLayer import WarpingLayer


class VTLNArtificiallyWarpedTrainer(VTLNSpeakerAdaptionModelTrainer):
    logger = logging.getLogger(__name__)

    #########################
    # Default constructor
    #
    def __init__(self, hparams):
        """Set your parameters here."""
        dir_world_labels = os.path.join(hparams.work_dir, "VTLNArtificiallyWarped/phoneme_alpha")
        dir_question_labels = os.path.join(hparams.work_dir, "questions")

        # Read which files to process.
        with open(os.path.join(hparams.data_dir, "file_id_list_" + hparams.voice + "_adapt_p276.txt")) as f:
            id_list = f.readlines()
        id_list[:] = [s.strip(' \t\n\r') for s in id_list]  # Trim line endings in-place.

        # TODO: Ignore unvoiced frames?
        # if hparams.add_deltas:
        #     self.loss_function = WMSELoss(hparams.num_coded_sps * 3 + 7, -4, weight=0.0, decision_index_weight=1.0, reduce=False)
        # else:
        #     self.loss_function = WMSELoss(hparams.num_coded_sps + 3, -2, weight=0.0, decision_index_weight=1.0, reduce=False)
        super().__init__(dir_world_labels, dir_question_labels, id_list, hparams.num_questions, hparams)

    def gen_figure_from_output(self, id_name, label, hidden, hparams):
        _, alphas = hidden
        labels_post = self.OutputGen.postprocess_sample(label)
        coded_sp, lf0, vuv, bap = WorldFeatLabelGen.convert_to_world_features(labels_post, contains_deltas=False,
                                                                              num_coded_sps=hparams.num_coded_sps)
        sp = WorldFeatLabelGen.mgc_to_sp(coded_sp, hparams.synth_fs)
        lf0, _ = interpolate_lin(lf0)

        # Load original lf0.
        org_labels_post = WorldFeatLabelGen.load_sample(id_name, self.OutputGen.dir_labels,
                                                        add_deltas=self.OutputGen.add_deltas,
                                                        num_coded_sps=hparams.num_coded_sps)
        original_mgc, original_lf0, original_vuv, *_ = WorldFeatLabelGen.convert_to_world_features(org_labels_post,
                                                                                                   contains_deltas=self.OutputGen.add_deltas,
                                                                                                   num_coded_sps=hparams.num_coded_sps)
        original_lf0, _ = interpolate_lin(original_lf0)

        np.random.seed(7)
        phonemes_to_alpha_tensor = ((np.random.rand(53) - 0.5) * 2 * 0.2)
        questions = QuestionLabelGen.load_sample(id_name, "experiments/" + hparams.voice + "/questions/",
                                                 num_questions=hparams.num_questions)[:len(alphas)]
        phoneme_indices = QuestionLabelGen.questions_to_phoneme_indices(questions, self.phoneme_indices)
        alpha_vec = phonemes_to_alpha_tensor[phoneme_indices]

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
        graphs.append((original_lf0, 'Original LF0'))
        graphs.append((lf0, 'NN LF0'))
        plotter.set_data_list(grid_idx=grid_idx, data_list=graphs)
        plotter.set_area_list(grid_idx=grid_idx, area_list=[(np.invert(vuv.astype(bool)), '0.8', 1.0),
                                                            (np.invert(original_vuv.astype(bool)), 'red', 0.2)])

        # grid_idx += 1
        # plotter.set_label(grid_idx=grid_idx, xlabel='frames [' + str(hparams.frame_size_ms) + ' ms]', ylabel='Original spectrogram')
        # plotter.set_specshow(grid_idx=grid_idx, spec=WorldFeatLabelGen.mgc_to_sp(original_mgc, hparams.synth_fs))
        #
        # grid_idx += 1
        # plotter.set_label(grid_idx=grid_idx, xlabel='frames [' + str(hparams.frame_size_ms) + ' ms]', ylabel='NN spectrogram')
        # plotter.set_specshow(grid_idx=grid_idx, spec=sp)

        grid_idx += 1
        plotter.set_label(grid_idx=grid_idx, xlabel='frames [' + str(hparams.frame_size_ms) + ' ms]', ylabel='alpha')
        graphs = list()
        graphs.append((alpha_vec, 'Original alpha'))
        graphs.append((alphas, 'NN alpha'))
        plotter.set_data_list(grid_idx=grid_idx, data_list=graphs)
        plotter.set_area_list(grid_idx=grid_idx, area_list=[(np.invert(vuv.astype(bool)), '0.8', 1.0),
                                                            (np.invert(original_vuv.astype(bool)), 'red', 0.2)])
        if hasattr(hparams, "phoneme_indices") and hparams.phoneme_indices is not None \
           and hasattr(hparams, "question_file") and hparams.question_file is not None:
            questions = QuestionLabelGen.load_sample(id_name,
                                                     "experiments/" + hparams.voice + "/questions/",
                                                     num_questions=hparams.num_questions)[:len(lf0)]
            np_phonemes = QuestionLabelGen.questions_to_phonemes(questions, hparams.phoneme_indices, hparams.question_file)
            plotter.set_annotations(grid_idx, np_phonemes)

        plotter.gen_plot()
        plotter.save_to_file(filename + '.VTLN' + hparams.gen_figure_ext)

    def compute_score(self, dict_outputs_post, dict_hiddens, hparams):
        mcd, f0_rmse, vuv_error_rate, bap_mcd = super().compute_score(dict_outputs_post, dict_hiddens, hparams)

        # Get data for comparision.
        dict_original_post = dict()
        for id_name in dict_outputs_post.keys():
            dict_original_post[id_name] = WorldFeatLabelGen.load_sample(id_name, self.OutputGen.dir_labels, True,
                                                                        num_coded_sps=hparams.num_coded_sps)

        # Create a warping layer for manual warping.
        wl = WarpingLayer((hparams.num_coded_sps,), (hparams.num_coded_sps,), hparams)
        if hparams.use_gpu:
            wl = wl.cuda()
        wl.set_norm_params(*self.OutputGen.norm_params)
        batch_size = len(dict_outputs_post)

        np.random.seed(7)
        phonemes_to_alpha_tensor = ((np.random.rand(53) - 0.5) * 2 * 0.2)

        for cep_coef_start in [0, 1]:
            for cep_coef_end in (range(10, 19) if cep_coef_start == 1 else [-1]):
                alphas_rmse = 0.0
                org_to_warped_mcd = 0.0
                org_to_nn_warping_mcd = 0.0
                output_to_warped_mcd = 0.0

                for id_name, labels in dict_outputs_post.items():
                    # Split NN output.
                    _, output_alphas = dict_hiddens[id_name]
                    output_mgc_post, *_ = self.OutputGen.convert_to_world_features(labels,
                                                                                   False,
                                                                                   num_coded_sps=hparams.num_coded_sps)

                    # Load the original sample without warping.
                    org_output = self.OutputGen.load_sample(id_name, "experiments/" + hparams.voice + "/WORLD/",
                                                            add_deltas=True, num_coded_sps=hparams.num_coded_sps)
                    org_output = org_output[:len(output_mgc_post)]
                    org_mgc_post = org_output[:, :hparams.num_coded_sps]
                    org_output_pre = self.OutputGen.preprocess_sample(org_output)  # Preprocess the sample.
                    org_mgc_pre = org_output_pre[:, :hparams.num_coded_sps * (3 if hparams.add_deltas else 1)]

                    # Load the original warped sample.
                    org_mgc_warped_post = dict_original_post[id_name][:len(output_mgc_post), :hparams.num_coded_sps]
                    # org_mgc_warped_post = self.OutputGen.load_sample(id_name, "experiments/" + hparams.voice + "/vtln_speaker_static/alpha_1.10/", add_deltas=True, num_coded_sps=hparams.num_coded_sps)[:len(output_mgc_post), :hparams.num_coded_sps]

                    # Compute error between warped version and NN output.
                    output_to_warped_mcd += metrics.melcd(org_mgc_warped_post[:, cep_coef_start:cep_coef_end],
                                                          output_mgc_post[:, cep_coef_start:cep_coef_end])
                    # Compute error between warped version and original one.
                    org_to_warped_mcd += metrics.melcd(org_mgc_warped_post[:, cep_coef_start:cep_coef_end],
                                                       org_mgc_post[:, cep_coef_start:cep_coef_end])

                    # Get original alphas from phonemes.
                    questions = QuestionLabelGen.load_sample(id_name, "experiments/" + hparams.voice + "/questions/",
                                                             num_questions=hparams.num_questions)[:len(output_alphas)]
                    phoneme_indices = QuestionLabelGen.questions_to_phoneme_indices(questions, self.phoneme_indices)
                    org_alphas = phonemes_to_alpha_tensor[phoneme_indices, None]

                    # Compute RMSE of alphas.
                    alphas_rmse += math.sqrt(((org_alphas - output_alphas) ** 2).sum())

                    # Warp the original mgcs with the alpha predicted by the network.
                    org_mgc_nn_warped, _ = wl(torch.from_numpy(org_mgc_pre[:, None, ...]), None, (len(org_mgc_pre),),
                                              (len(org_mgc_pre),), alphas=torch.from_numpy(output_alphas[:, None, ...]))  # Warp with the NN alphas.
                    org_output_pre[:, :hparams.num_coded_sps * (3 if hparams.add_deltas else 1)] = org_mgc_nn_warped[:, 0, ...].detach()  # Write warped mgcs back.
                    org_mgc_nn_warped_post = self.OutputGen.postprocess_sample(org_output_pre, apply_mlpg=False)[:, :hparams.num_coded_sps]  # Postprocess sample.

                    # Compute error between correctly warped version and original mgcs warped with NN alpha.
                    org_to_nn_warping_mcd += metrics.melcd(org_mgc_warped_post[:, cep_coef_start:cep_coef_end],
                                                           org_mgc_nn_warped_post[:, cep_coef_start:cep_coef_end])

                alphas_rmse /= batch_size
                output_to_warped_mcd /= batch_size
                org_to_warped_mcd /= batch_size
                org_to_nn_warping_mcd /= batch_size

                self.logger.info("MCep from {} to {}:".format(cep_coef_start, cep_coef_end))
                self.logger.info("RMSE alphas: {:4.2f}".format(alphas_rmse))
                self.logger.info("Original mgc to warped mgc error: {:4.2f}dB".format(org_to_warped_mcd))
                self.logger.info("Original mgc warped by network alpha to warped mgc error: {:4.2f}dB ({:2.2f}%)".format(org_to_nn_warping_mcd, (1 - org_to_nn_warping_mcd / org_to_warped_mcd) * 100))
                self.logger.info("Network output to original warped mgc error: {:4.2f}dB".format(output_to_warped_mcd))

        return mcd, f0_rmse, vuv_error_rate, bap_mcd


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

        # Read which files to process.
        with open(os.path.join(hparams.data_dir, "file_id_list_" + hparams.voice + "_adapt_p276.txt")) as f:
            id_list = f.readlines()
        id_list = [s.strip(' \t\n\r') for s in id_list]  # Trim line endings in-place.

        self.loss_function = WMSELoss(hparams.num_coded_sps * 3 + 7, -4, weight=0.0, decision_index_weight=1.0, reduce=False)
        super().__init__(dir_world_labels, dir_question_labels, id_list, hparams.num_questions, hparams)


def main():
    logging.basicConfig(level=logging.INFO)

    hparams = VTLNArtificiallyWarpedTrainer.create_hparams()  # TODO: Parse input for hparams.

    # General parameters.
    hparams.phoneme_indices = np.arange(59, 107, dtype=np.int)
    hparams.question_file = "../../../tools/tts_frontend/questions/questions-en-radio_dnn_416.hed"
    # hparams.phoneme_indices = np.arange(86, 347, 5, dtype=np.int)
    # hparams.question_file = "../../../tools/tts_frontend/questions/questions-de-festival_496.hed"
    hparams.num_questions = 425
    hparams.voice = "English"
    hparams.work_dir = os.path.realpath(os.path.join("experiments", hparams.voice))
    hparams.data_dir = os.path.realpath("database")
    hparams.out_dir = os.path.join(hparams.work_dir, "VTLNArtificiallyWarped")
    hparams.num_speakers = 1
    hparams.speaker_emb_dim = 16
    hparams.sampling_frequency = 16000
    hparams.frame_size_ms = 5
    hparams.seed = 1234
    hparams.num_coded_sps = 30
    hparams.add_deltas = True

    # Training parameters.
    hparams.epochs = 25
    hparams.use_gpu = True
    hparams.dropout = 0.05
    hparams.batch_size_train = 32
    hparams.batch_size_val = 48
    hparams.batch_size_benchmark = 48
    hparams.grad_clip_norm_type = 2
    hparams.grad_clip_max_norm = 100
    hparams.use_saved_learning_rate = True
    hparams.optimiser_args["lr"] = 0.001
    hparams.optimiser_type = "Adam"
    hparams.scheduler_type = "Plateau"
    hparams.scheduler_args["patience"] = 5
    hparams.start_with_test = True
    hparams.epochs_per_checkpoint = 5
    hparams.save_final_model = True
    hparams.use_best_as_final_model = True

    # hparams.model_type = None
    hparams.model_type = "RNNDYN-2_RELU_1024-3_BiLSTM_512-1_FC_97"  # Average model.
    hparams.f_get_emb_index = None  # No embedding input for average model.

    hparams.model_name = "Bds-p276{}-lr{}.nn".format("-dropout" + str(hparams.dropout).split('.')[1] if hparams.dropout != 0 else "",
                                                     str(hparams.optimiser_args["lr"]).split('.')[1])

    trainer = BaselineTrainer(hparams)
    trainer.init(hparams)
    trainer.train(hparams)
    trainer.benchmark(hparams)

    # hparams.synth_gen_figure = False
    hparams.synth_vocoder = "WORLD"

    synth_list = dict()
    synth_list["train"] = ["p276/p276_020"]
    synth_list["val"] = ["p276/p276_010"]
    synth_list["test"] = ["p276/p276_030", "p276/p276_160"]
    # for key, value in synth_list.items():
    #     hparams.synth_file_suffix = "_" + str(key) + "_" + hparams.synth_vocoder
    #     trainer.synth(hparams, synth_list[key])
    #     # trainer.gen_figure(synth_list[key], hparams)

    # Training parameters.
    hparams.epochs = 15
    hparams.train_pre_net = False
    hparams.dropout = 0.05
    hparams.batch_size_train = 2
    hparams.batch_size_val = hparams.batch_size_train
    hparams.batch_size_benchmark = hparams.batch_size_train

    hparams.pre_net_model_name = hparams.model_name  # Use the just trained baseline model.
    # hparams.model_type = None
    hparams.model_type = "VTLN"
    hparams.model_name = "VTLN-phoneme_alpha.nn"
    hparams.pass_embs_to_pre_net = False
    hparams.f_get_emb_index = (lambda id_name, length: np.zeros((length, 1)),)  # Model with only one speaker, which is the artificial one.

    # Training.
    trainer = VTLNArtificiallyWarpedTrainer(hparams)
    trainer.init(hparams)
    trainer.train(hparams)
    trainer.benchmark(hparams)

    hparams.synth_gen_figure = True
    for key, value in synth_list.items():
        hparams.synth_file_suffix = "_" + str(key) + "_" + hparams.synth_vocoder
        trainer.synth(hparams, synth_list[key])
        # trainer.gen_figure(synth_list[key], hparams)


if __name__ == "__main__":
    main()
