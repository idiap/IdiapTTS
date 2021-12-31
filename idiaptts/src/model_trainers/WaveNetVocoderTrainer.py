#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#

"""Module description:
   Train a model to generate raw audio from WORLD features (60 MFCC, 1 LF0, 1 VUV, 1 BAP).
"""

# System imports.
import copy
from functools import partial, reduce
import logging
from operator import mul
import os
from typing import List

# Third-party imports.
import numpy as np

# Local source tree imports.
from idiaptts.misc.utils import sample_linearly
from idiaptts.src.DataPlotter import DataPlotter
from idiaptts.src.data_preparation.audio.RawWaveformLabelGen import RawWaveformLabelGen
from idiaptts.src.data_preparation.DataReaderConfig import DataReaderConfig
from idiaptts.src.data_preparation.PyTorchDatareadersDataset import PyTorchDatareadersDataset
from idiaptts.src.data_preparation.PyTorchLabelGensDataset import PyTorchLabelGensDataset as LabelGensDataset
from idiaptts.src.data_preparation.world.WorldFeatLabelGen import WorldFeatLabelGen
from idiaptts.src.ExtendedHParams import ExtendedHParams
from idiaptts.src.model_trainers.ModularTrainer import ModularTrainer
from idiaptts.src.neural_networks.pytorch.loss.OneHotCrossEntropyLoss import OneHotCrossEntropyLoss
from idiaptts.src.neural_networks.pytorch.loss.DiscretizedMixturelogisticLoss import DiscretizedMixturelogisticLoss
from idiaptts.src.neural_networks.pytorch.ModelHandlerPyTorch import ModelHandlerPyTorch as ModelHandler
from idiaptts.src.neural_networks.pytorch.models.WaveNetWrapper import WaveNetWrapper


class WaveNetVocoderTrainer(ModularTrainer):
    logger = logging.getLogger(__name__)

    #########################
    # Default constructor
    #
    def __init__(self, hparams: ExtendedHParams, id_list: List[str],
                 data_reader_configs: List[DataReaderConfig] = None):

        if hparams is None:
            hparams = self.create_hparams()
            hparams.out_dir = os.path.curdir

        # Write missing default parameters.
        # if hparams.synth_dir is None:
        #     hparams.synth_dir = os.path.join(hparams.out_dir, "synth")

        super().__init__(data_reader_configs=data_reader_configs,
                         id_list=id_list,
                         hparams=hparams)

        # in_to_out_multiplier = int(hparams.frame_rate_output_Hz /
        #                            (1000.0 / hparams.frame_size_ms))
        # num_frames_per_sec = 1000.0 / hparams.frame_size_ms
        # # NOTE: Had to limit input length because of memory constraints.
        # max_frames_trainset = int(num_frames_per_sec * hparams.max_input_train_sec) \
        #     * in_to_out_multiplier  # Multiply by number of seconds.
        # max_frames_testset = int(num_frames_per_sec * hparams.max_input_test_sec) \
        #     * in_to_out_multiplier  # Ensure that test takes all frames.

        # self.dataset_train = LabelGensDataset(
        #     self.id_list_train, self.InputGen, self.OutputGen, hparams,
        #     random_select=True, max_frames_input=max_frames_trainset)
        # self.dataset_val = LabelGensDataset(
        #     self.id_list_val, self.InputGen, self.OutputGen, hparams,
        #     random_select=True, max_frames_input=max_frames_testset)

        # if self.loss_function is None:
        #     if hparams.input_type == "mulaw-quantize":
        #         self.loss_function = OneHotCrossEntropyLoss(reduction='none', shift=1)
        #     else:
        #         self.loss_function = DiscretizedMixturelogisticLoss(hparams.quantize_channels,
        #                                                             hparams.log_scale_min,
        #                                                             reduction='none',
        #                                                             hinge_loss=hparams.hinge_regularizer)

        if hparams.scheduler_type == "default":
            hparams.scheduler_type = "Noam"
            # hparams.scheduler_args['exponential_gamma'] = 0.99
            hparams.scheduler_args['wormup_steps'] = 4000

        # Override the collate and decollate methods of batches.
        self.batch_collate_fn = partial(self.prepare_batch, use_cond=hparams.use_cond, one_hot_target=True)
        self.batch_decollate_fn = self.decollate_network_output

    def _setup_datareaders(self, data_reader_configs: List[DataReaderConfig],
                           hparams: ExtendedHParams):
        super()._setup_datareaders(data_reader_configs=data_reader_configs,
                                   hparams=hparams)

        in_to_out_multiplier = int(hparams.frame_rate_output_Hz /
                                   (1000.0 / hparams.frame_size_ms))
        num_frames_per_sec = 1000.0 / hparams.frame_size_ms
        # NOTE: Had to limit input length because of memory constraints.
        max_frames_trainset = int(num_frames_per_sec * hparams.max_input_train_sec) \
            * in_to_out_multiplier  # Multiply by number of seconds.
        max_frames_testset = int(num_frames_per_sec * hparams.max_input_test_sec) \
            * in_to_out_multiplier  # Ensure that test takes all frames.

        readers = list(self.datareaders.values())
        for reader in readers:
            reader.max_frames = max_frames_trainset
        self.dataset_val = PyTorchDatareadersDataset(
            id_list=self.id_list_val,
            datareaders=readers,
            hparams=hparams)

        readers = copy.deepcopy(readers)
        for reader in readers:
            reader.max_frames = max_frames_testset
        self.dataset_val = PyTorchDatareadersDataset(
            id_list=self.id_list_val,
            datareaders=readers,
            hparams=hparams)

    @staticmethod
    def legacy_support_init(dir_world_features: os.PathLike, id_list: List[str],
                            hparams: ExtendedHParams):
        in_to_out_multiplier = int(hparams.frame_rate_output_Hz /
                                   (1000.0 / hparams.frame_size_ms))
        # num_frames_per_sec = 1000.0 / hparams.frame_size_ms
        # NOTE: Had to limit input length because of memory constraints.
        # max_frames = int(num_frames_per_sec * hparams.max_input_train_sec) \
        #     * in_to_out_multiplier  # Multiply by number of seconds.
        # max_frames_testset = int(num_frames_per_sec * hparams.max_input_test_sec) \
        #     * in_to_out_multiplier  # Ensure that test takes all frames.

        if hparams.input_type == WaveNetWrapper.Config.INPUT_TYPE_MULAW:
            mu = hparams.mu
        else:
            mu = None

        data_reader_configs = [
            DataReaderConfig(
                name="mfbanks",
                directory=dir_world_features,
                feature_type="WorldFeatLabelGen",
                features=[hparams.sp_type + str(hparams.num_coded_sps)],
                output_names=["conditioning"],
                requires_seq_mask=True,
                random_select=True,
                # max_frames=max_frames,
                load_sp=hparams.load_sp,
                load_lf0=hparams.load_lf0,
                load_vuv=hparams.load_vuv,
                load_bap=hparams.load_bap,
                match_length=["questions"],
                add_deltas=False,
                num_coded_sps=hparams.num_coded_sps,
                num_bap=hparams.num_bap,
                sp_type=hparams.sp_type,
                preprocessing_fn=partial(sample_linearly,
                                         in_to_out_multiplier=in_to_out_multiplier,
                                         dtype=np.float32),
            ),
            DataReaderConfig(
                name="target",
                feature_type="RawWaveformLabelGen",
                directory=hparams.data_dir,
                frame_rate_output_Hz=hparams.frame_rate_output_Hz,
                frame_size_ms=hparams.frame_size_ms,
                mu=mu,
                silence_threshold_quantized=hparams.silence_threshold_quantized,
                requires_seq_mask=True
            )
        ]

        return dict(data_reader_configs=data_reader_configs, hparams=hparams,
                    id_list=id_list)

    @staticmethod
    def create_hparams(hparams_string=None, verbose=False):
        """Create model hyper-parameters. Parse non-default from given string."""
        hparams = ModularTrainer.create_hparams(hparams_string, verbose=False)
        hparams.synth_vocoder = "raw"

        hparams.add_hparams(
            batch_first=True,
            frame_rate_output_Hz=16000,
            bit_depth=16,
            silence_threshold_quantized=None,  # Beginning and end of audio below the threshold are trimmed.
            teacher_forcing_in_test=True,
            ema_decay=0.9999,
            mu=255,

            # Model parameters.
            input_type=WaveNetWrapper.Config.INPUT_TYPE_MULAW,
            # hinge_regularizer=True,  # Only used in MoL prediction (input_type="raw").
            # log_scale_min=float(np.log(1e-14)),  # Only used for mixture of logistic distributions.
            # quantize_channels=256
            )  # 256 for input type mulaw-quantize, otherwise 65536
        # if hparams.input_type == "mulaw-quantize":
        #     hparams.add_hparam("out_channels", hparams.quantize_channels)
        # else:
        #     hparams.add_hparam("out_channels", 10 * 3)  # num_mixtures * 3 (pi, mean, log_scale)

        hparams.add_hparams(
            # layers=24,  # 20
            # stacks=4,  # 2
            # residual_channels=512,
            # gate_channels=512,
            # skip_out_channels=256,
            # dropout=1 - 0.95,
            # kernel_size=3,
            # weight_normalization=True,
            use_cond=True,  # Determines if conditioning is used.
            # cin_channels=63,
            # upsample_conditional_features=False,
            # upsample_scales=[
            #     5,
            #     4,
            #     2
            # ]
            )
        if hparams.has_value("upsample_conditional_features"):
            hparams.len_in_out_multiplier = reduce(mul, hparams.upsample_scales, 1)
        else:
            hparams.len_in_out_multiplier = 1

        hparams.add_hparams(
            # freq_axis_kernel_size=3,
            # gin_channels=-1,
            # n_speakers=1,
            # use_speaker_embedding=False,
            sp_type="mfbanks",
            load_sp=True,
            load_lf0=False,
            load_vuv=False,
            load_bap=False
        )

        if verbose:
            logging.info(hparams.get_debug_string())

        return hparams

    # Load train and test data.
    @staticmethod
    def prepare_batch(batch, common_divisor=1, batch_first=False, use_cond=True, one_hot_target=True):
        inputs, targets, seq_lengths_input, seq_lengths_output, mask, permutation = ModelHandler.prepare_batch(batch, common_divisor=common_divisor, batch_first=batch_first)

        if batch_first:
            # inputs: (B x T x C) --permute--> (B x C x T)
            inputs = inputs.transpose(1, 2).contiguous()
        # TODO: Handle case where batch_first=False: inputs = inputs.transpose(2, 0, 1).contiguous()?

        if targets is not None:
            if batch_first:
                # targets: (B x T x C) --permute--> (B x C x T)
                targets = targets.transpose(1, 2).contiguous()

            if not one_hot_target:
                targets = targets.max(dim=1, keepdim=True)[1].float()

        if mask is not None:
            mask = mask[:, 1:].contiguous()

        return inputs if use_cond else None, targets, seq_lengths_input, seq_lengths_output, mask, permutation

    @staticmethod
    def decollate_network_output(output, hidden, seq_lengths=None, permutation=None, batch_first=True):

        # Output of r9y9 Wavenet has batch first, thus output: B x C x T --transpose--> B x T x C
        output = np.transpose(output, (0, 2, 1))
        if not batch_first:
            # output: B x T x C --transpose--> T x B x C
            output = np.transpose(output, (1, 0, 2))
        return ModelTrainer.split_batch(output, hidden, seq_length_output=seq_lengths, permutation=permutation, batch_first=batch_first)

    def gen_figure_from_output(self, id_name, labels, hidden, hparams):

        labels_post = self.dataset_train.postprocess_sample(labels)  # Labels come in as T x C.
        org_raw = RawWaveformLabelGen.load_sample(id_name, self.OutputGen.frame_rate_output_Hz)

        # Get a data plotter.
        plotter = DataPlotter()
        net_name = os.path.basename(hparams.model_name)
        id_name = os.path.basename(id_name).rsplit('.', 1)[0]
        filename = os.path.join(hparams.out_dir, id_name + "." + net_name)
        plotter.set_title(id_name + " - " + net_name)
        grid_idx = 0

        graphs = list()
        graphs.append((org_raw, 'Org'))
        graphs.append((labels_post, 'Wavenet'))
        plotter.set_data_list(grid_idx=grid_idx, data_list=graphs)
        plotter.set_linewidth(grid_idx=grid_idx, linewidth=[0.1])
        plotter.set_colors(grid_idx=grid_idx, alpha=0.8)
        plotter.set_lim(grid_idx, ymin=-1, ymax=1)
        plotter.set_label(grid_idx=grid_idx, xlabel='frames [' + str(hparams.frame_rate_output_Hz) + ' Hz]', ylabel='raw')

        plotter.gen_plot()
        plotter.save_to_file(filename + '.Raw' + hparams.gen_figure_ext)

    def save_for_vocoding(self, filename):
        # Save the full model so that hyper-parameters are already set.
        self.model_handler.save_full_model(filename, self.model_handler.model, verbose=True)
        # Save an easily loadable version of the normalisation parameters on the input side used during training.
        np.save(os.path.splitext(filename)[0] + "_norm_params", np.concatenate(self.InputGen.norm_params, axis=0))
