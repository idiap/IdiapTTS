#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#

"""Module description:
"""

# System imports.
import logging
import os
from typing import List

# Third-party imports.

# Local source tree imports.
from idiaptts.src.model_trainers.AcousticModelTrainer import AcousticModelTrainer
from idiaptts.src.neural_networks.pytorch.ModelHandlerPyTorch import ModelHandlerPyTorch
from idiaptts.src.neural_networks.pytorch.ModularModelHandlerPyTorch import ModularModelHandlerPyTorch
from idiaptts.src.data_preparation.DataReaderConfig import DataReaderConfig
from idiaptts.src.ExtendedHParams import ExtendedHParams


class EncDecMonophoneModelTrainer(AcousticModelTrainer):  # TODO: Do not inherit from AcousticModelTrainer just to get its gen_figure_from_output function.
    """
    Implementation of a ModelTrainer for the generation of acoustic data.

    Use phonemes without duration as input and predict acoustic features.
    """
    logger = logging.getLogger(__name__)

    #########################
    # Default constructor
    #
    def __init__(self, hparams: ExtendedHParams, id_list: List[str]):
        if hparams is None:
            hparams = self.create_hparams()
            hparams.out_dir = os.path.curdir

        # super(AcousticModelTrainer, self).__init__(id_list=id_list, data_reader_configs=data_reader_configs, hparams=hparams)  # Call ModelTrainer base class init directly.
        super().__init__(id_list=id_list, hparams=hparams)

        if hparams.scheduler_type == "default":
            hparams.scheduler_type = "Plateau"
            hparams.scheduler_args["verbose"] = True
            # hparams.scheduler_type = "DelayedExponential"
            # hparams.scheduler_args["wormup_steps"] = 50000
            # hparams.scheduler_args["gamma"] = 0.999
            # hparams.iterations_per_scheduler_step = 100

        # self.batch_collate_fn = self.prepare_batch

    def legacy_support_init(self, dir_world_features, dir_monophone_labels, dir_durations, id_list,
                            file_symbol_dict, hparams=None):

        datareader_configs = []
        datareader_configs.append(
            DataReaderConfig(
                name="phonemes",
                feature_type="PhonemeLabelGen",
                directory=dir_monophone_labels,
                features="phonemes",
                file_symbol_dict=file_symbol_dict,
                label_type="label_state_align",
                one_hot=False
            )
        )
        hparams.n_symbols = datareader_configs[0].num_symbols

        datareader_configs.append(
            DataReaderConfig(
                name="cmp_features",
                feature_type="WorldFeatLabelGen",
                directory=dir_world_features,
                features=["cmp_mcep" + str(hparams.num_coded_sps)],
                output_names=["acoustic_features"],
                chunk_size=hparams.n_frames_per_step,
                add_deltas=hparams.add_deltas,
                num_coded_sps=hparams.num_coded_sps,
                sp_type=hparams.sp_type,
                load_sp=hparams.load_sp,
                load_lf0=hparams.load_lf0,
                load_vuv=hparams.load_vuv,
                load_bap=hparams.load_bap,
                requires_seq_mask=True
            )
        )

        datareader_configs.append(
            DataReaderConfig(
                name="durations",
                feature_type="PhonemeDurationLabelGen",
                directory=dir_durations,
                features="durations",
                output_names=["attention_matrix"],
                match_length=["cmp_features", "phonemes"],
                load_as_matrix=True
            )
        )

        return dict(data_reader_configs=datareader_configs, hparams=hparams, id_list=id_list)

    # Load train and test data.
    @staticmethod
    def prepare_batch(batch, common_divisor=1, batch_first=False, add_EOF_gate=True, fixed_attention=True, n_frames_per_step=1):
        inputs, targets, seq_lengths_input, seq_lengths_output, mask, permutation, *extra_labels = ModularModelHandlerPyTorch.prepare_batch(batch, common_divisor, batch_first)

        raise NotImplementedError()
        if targets is not None:
            num_chunks = int(targets.shape[1 if batch_first else 0] / n_frames_per_step)
            max_length = num_chunks * n_frames_per_step
            if batch_first:
                targets = targets[:, :max_length]
                seq_lengths_output.clamp_(max=max_length)
                mask = mask[:, :max_length]
            else:
                targets = targets[:max_length]
                seq_lengths_output.clamp_(max=max_length)
                mask = mask[:max_length]
            targets = targets.contiguous()
            mask = mask.contiguous()

        if fixed_attention:
            if targets is None:
                seq_lengths_output = torch.tensor(list(map(len, extra_labels[0])), dtype=torch.long)
                num_chunks = int(seq_lengths_output.max() / n_frames_per_step)
                max_length = num_chunks * n_frames_per_step
                seq_lengths_output.clamp_(max=max_length)

            attention_matrices = extra_labels[0]

            # B = len(attention_matrices)
            attentions = list()
            max_P = seq_lengths_input.max()
            max_T = seq_lengths_output.max()
            for batch_idx, (attention_matrix, P, target_T) in enumerate(zip(attention_matrices, seq_lengths_input, seq_lengths_output)):
                # Attention matrix must be (max P x max T)
                # P = attention_matrix.shape[0]
                T = attention_matrix.shape[0]  # Attention and target T have a slight miss match.
                padded_attention_matrix = torch.nn.ConstantPad2d((0, max_P - P, 0, max_T - T), 0.0)(attention_matrix)
                padded_attention_matrix[T:target_T, P-1] = 1.0  # This compensates the small length miss match.
                attentions.append(padded_attention_matrix)
            attentions = torch.stack(attentions, dim=0 if batch_first else 1)
            extra_labels[0] = attentions
        else:
            extra_labels = extra_labels[1:] if len(extra_labels) > 1 else []

        if targets is not None:
            if add_EOF_gate:
                targets, mask = EncDecMonophoneModelTrainer._add_eof_gate(targets, mask, seq_lengths_output, batch_first)

        return (inputs, targets, seq_lengths_input, seq_lengths_output, mask, permutation, *extra_labels)

    @staticmethod
    def _add_eof_gate(targets, mask, seq_lengths_output, batch_first):
        if batch_first:
            batch_size = targets.shape[0]
            target_gate = torch.zeros((batch_size, seq_lengths_output.max(), 1), dtype=torch.float32)
            for index, length in enumerate(seq_lengths_output):
                target_gate[index, length - 1:] = 1.0
        else:
            batch_size = targets.shape[1]
            target_gate = torch.zeros((seq_lengths_output.max(), batch_size, 1), dtype=torch.float32)
            for index, length in enumerate(seq_lengths_output):
                target_gate[length - 1:, index] = 1.0

        if mask is not None:
            mask_gate = torch.tensor([1.0], dtype=torch.float32).expand(mask.shape[:-1]).unsqueeze(-1)
            mask = torch.cat((mask.expand(-1, -1, targets.shape[-1]), mask_gate), dim=-1)

        targets = torch.cat((targets, target_gate), dim=-1)
        # targets_per = None
        # targets_per = targets.transpose_(0, 2).contiguous()  #.transpose(0, 2) # .permute(1, 2, 0).contiguous()
        # targets_per = targets.permute(1, 2, 0).contiguous()  # B x C x T

        #targets_1 = targets_per[:-1, :, :]
        #targets_1c = targets_1.contiguous()
        #logging.info(targets_1 is targets_1c)

        return targets, mask

    @staticmethod
    def create_hparams(hparams_string=None, verbose=False):

        hparams = AcousticModelTrainer.create_hparams(hparams_string, verbose=False)
        hparams.add_deltas = False

        default_hparams = {
            # "n_output_channels": None,
            "load_sp": True,
            "load_lf0": True,
            "load_vuv": True,
            "load_bap": True,

            "n_symbols": None,
            "symbols_embedding_dim": 512,

            # # Encoder parameters
            #
            # # Decoder parameters
            "add_EOF_gate": False,
            "n_frames_per_step": 1,
            "single_encoder_input_per_step": False,
            "max_decoder_steps": 1000,  # 200*18,
            # "gate_threshold": 0.5,
            "p_text_encoder_dropout": 0.5,
            "p_audio_encoder_dropout": 0.5,
            # "p_attention_dropout": 0.4,
            "p_decoder_dropout": 0.1,
            "p_audio_decoder_dropout": 0.5,
            #
            # # Attention parameters
            # "attention_rnn_dim": 1024,
            # "attention_dim": 128,
            #
            # # Location Layer parameters
            # "attention_location_n_filters": 32,
            # "attention_location_kernel_size": 31,
            #
            # # Mel-post processing network parameters
            # "postnet_embedding_dim": 512,
            # "postnet_kernel_size": 5,
            # "postnet_n_convolutions": 5,
            #
            # "mask_padding": False,
        }

        # Fill missing values by default values.
        for key, value in default_hparams.items():
            if not hasattr(hparams, key):
                hparams.add_hparam(key, value)

        if verbose:
            logging.info('Final parsed hparams: %s', hparams.values())

        return hparams

    def compute_score(self, *args, **kwargs):
        raise NotImplementedError("Not clear what objective measure to use.")