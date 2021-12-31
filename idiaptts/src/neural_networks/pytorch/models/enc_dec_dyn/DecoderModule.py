#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#
import logging
import math

import numpy as np
import torch

from .SubModule import SubModule
from . import FIXED_ATTENTION, ATTENTION_GROUND_TRUTH
import idiaptts.src.neural_networks.pytorch.models.rnn_dyn as rnn_dyn
from idiaptts.src.neural_networks.pytorch.models.ModelConfig import ModelConfig
from idiaptts.src.neural_networks.pytorch.models.NamedForwardModule import NamedForwardModule

from idiaptts.src.neural_networks.pytorch.models import enc_dec_dyn
from idiaptts.src.neural_networks.pytorch.models.enc_dec_dyn.attention.FixedAttention import FixedAttention


class DecoderModule(SubModule):

    def __init__(self, config):
        super().__init__(config)
        self.attention_args = config.attention_args
        self.attention_config = config.attention_config
        if config.attention_config == FIXED_ATTENTION:
            self.attention = enc_dec_dyn.FixedAttention.Config(
                config.attention_args[ATTENTION_GROUND_TRUTH],
                config.n_frames_per_step).create_model()
        self.input_merge_type = config.input_merge_type
        self.input_names = config.input_names
        self.n_frames_per_step = config.n_frames_per_step
        self.p_teacher_forcing = config.p_teacher_forcing
        self.teacher_forcing_input_names = config.teacher_forcing_input_names

        self.pre_net = None
        if config.pre_net_config is not None:
            # TODO: Allow multiple pre_nets same as projections.
            self.pre_net = config.pre_net_config.create_model()
        self.projections = []
        if config.projection_configs is not None:
            self.projections = [c.create_model() for c in config.projection_configs]
            for projector in self.projections:
                projector_id = "{}_{}".format(self.name, projector.name)
                logging.debug("Adding {} to {}".format(projector_id, self.name))
                self.add_module(projector_id, projector)

        self.decoder_dim = None
        if config.projection_configs is not None:
            projector_dims = [c.out_dim for c in config.projection_configs
                              if c.is_autoregressive_input]
            if len(projector_dims) > 0:
                self.decoder_dim = sum(projector_dims)

        # TODO: Not sure what I tried here.
        # elif self.pre_net is not None and hasattr(self.pre_net, "out_dim"):
        #     self.decoder_dim = self.pre_net.out_dim
        else:
            try:
                self.decoder_dim = self.model[-1].out_dim
            except (AttributeError, TypeError):
                if self.is_autoregressive \
                        or not type(self.attention) == FixedAttention:
                    raise ValueError("Cannot infer decoder output dimension.")

        self.decoder_output_name = getattr(config, "decoder_output_name", None)

    @property
    def is_autoregressive(self):
        return self.teacher_forcing_input_names is not None

    def init_hidden(self, batch_size=1):
        super().init_hidden(batch_size)
        if self.pre_net is not None:
            self.pre_net.init_hidden(batch_size)
        for projector in self.projections:
            projector.init_hidden(batch_size)

    def forward(self, data, lengths, max_lengths):
        input_ = NamedForwardModule._get_inputs(data, self.input_names,
                                                self.input_merge_type,
                                                batch_first=True)
        if self.is_autoregressive and all([name in data for name in
                                           self.teacher_forcing_input_names]):
            target = NamedForwardModule._get_inputs(
                data,
                self.teacher_forcing_input_names,
                ModelConfig.MERGE_TYPE_CAT,
                batch_first=True)
        else:
            target = None

        attention_matrix = None
        if type(self.attention) == FixedAttention:
            attention_matrix = NamedForwardModule._get_inputs(
                data,
                self.attention.attention_ground_truth,
                ModelConfig.MERGE_TYPE_LIST,
                batch_first=True)

        if self.attention.allows_batched_forward() \
            and (self.teacher_forcing_input_names is None
                 or (target is not None and self.p_teacher_forcing == 1.0)):
            output_dict, output_lengths, output_max_lengths = self._forward_batched(
                input_, target, lengths, max_lengths, attention_matrix)
        else:
            output_dict, output_lengths, output_max_lengths = self._forward_iterative(
                input_, target, lengths, max_lengths, attention_matrix)

        lengths.update(output_lengths)
        max_lengths.update(output_max_lengths)
        data.update(output_dict)

    def _forward_batched(self, input_, target=None, lengths=None,
                         max_lengths=None, attention_matrix=None):
        lengths = {k: v // self.n_frames_per_step for k, v in lengths.items()}
        max_lengths = {k: v // self.n_frames_per_step for k, v in max_lengths.items()}
        output_lengths = lengths[self.input_names[0]]
        output_max_lengths = max_lengths[self.input_names[0]]

        output_dict = dict()
        output_lengths_dict = {}
        output_max_lengths_dict = {}

        if attention_matrix is not None:
            attention_context, attention_matrix = self.attention.forward_batched(
                input_, attention_matrix)
            input_ = attention_context
            output_lengths = lengths[self.attention.attention_ground_truth]
            output_max_lengths = max_lengths[self.attention.attention_ground_truth]

            output_dict['attention'] = attention_matrix
            output_lengths_dict['attention'] = \
                lengths[self.attention.attention_ground_truth]
            output_max_lengths_dict['attention'] = \
                max_lengths[self.attention.attention_ground_truth]

        if self.is_autoregressive:
            teacher_forcing_target = self._get_teacher_forcing_target(target)
            target_lengths = lengths[self.teacher_forcing_input_names[0]]
            max_target_lengths = max_lengths[self.teacher_forcing_input_names[0]]

            if self.pre_net is not None:
                teacher_forcing_target, _ = self.pre_net(
                    teacher_forcing_target,
                    seq_lengths_input=target_lengths,
                    max_length_inputs=max_target_lengths)
            input_ = torch.cat((input_, teacher_forcing_target), dim=2)

        if self.model is None:
            decoder_output = input_
        elif type(self.model) is rnn_dyn.RNNDyn:
            if self.teacher_forcing_input_names is not None:
                length_ref_key = self.teacher_forcing_input_names[0]
            elif attention_matrix is not None:
                length_ref_key = self.attention.attention_ground_truth
            else:
                length_ref_key = self.input_names[0]
            decoder_output, kwargs = self.model(
                input_,
                seq_lengths_input=lengths[length_ref_key],
                max_length_inputs=max_lengths[length_ref_key])
            output_lengths = kwargs["seq_lengths_input"]
            output_max_lengths = kwargs["max_length_inputs"]
        else:
            decoder_output = self.forward_module(input_, lengths, max_lengths)

        if self.decoder_output_name is not None:
            output_dict[self.decoder_output_name] = decoder_output
            output_lengths_dict[self.decoder_output_name] = output_lengths
            output_max_lengths_dict[self.decoder_output_name] = output_max_lengths

        if len(self.projections) > 0:
            for projector in self.projections:
                projector_output = projector.forward_module(
                    decoder_output, lengths, max_lengths)
                projector_output = self._parse_outputs(projector_output)

                output_dict[projector.output_names[0]] = projector_output
                output_lengths_dict[projector.output_names[0]] = \
                    output_lengths * self.n_frames_per_step
                output_max_lengths_dict[projector.output_names[0]] = \
                    output_max_lengths * self.n_frames_per_step
        else:
            first_name = self.output_names[0]
            output_dict.update({first_name: decoder_output})
            output_lengths_dict[first_name] = output_lengths * self.n_frames_per_step
            output_max_lengths_dict[first_name] = output_max_lengths * self.n_frames_per_step

        return output_dict, output_lengths_dict, output_max_lengths_dict

    # def forward_fixed_attention_batched(self, input_, attention_matrix):
    #     B = attention_matrix.shape[0]
    #     T = attention_matrix.shape[1]
    #     num_chunks = int(T / self.n_frames_per_step)
    #     # attention_matrix = attention_matrix[:, ::self.n_frames_per_step]  # Skip frames.
    #     attention_matrix = attention_matrix.view(
    #         B, num_chunks, self.n_frames_per_step, -1).mean(dim=2)
    #     attention_context = torch.bmm(attention_matrix, input_)
    #     return attention_context

    def _get_teacher_forcing_target(self, target):
        shifted_target = target[:, self.n_frames_per_step - 1::self.n_frames_per_step][:, :-1]
        return torch.cat((self._get_go_frame(target), shifted_target), dim=1)

    def _get_go_frame(self, input):
        B = input.shape[0]
        return input.new_zeros(B, 1, self.decoder_dim)

    def _parse_outputs(self, decoder_outputs):
        B = decoder_outputs.shape[0]
        T = decoder_outputs.shape[1]
        return decoder_outputs.view(B, T * self.n_frames_per_step, -1)

    def _forward_iterative(self, input_, target, lengths, max_lengths,
                           attention_matrix):
        all_outputs_dict = {}
        all_attentions = []

        B = input_.shape[0]
        seq_lengths_ones = input_.new_ones(B, dtype=torch.long)

        if target is not None:
            max_steps = target.shape[1]
        elif attention_matrix is not None:
            max_steps = attention_matrix.shape[1]
        else:
            max_steps = self.max_decoder_steps

        decoder_input = self._get_go_frame(input_)
        if attention_matrix is None:
            attention_state = self.attention.get_go_frame()
        else:
            attention_state = attention_matrix

        for i in range(math.ceil(max_steps / self.n_frames_per_step)):
            if self.is_autoregressive:
                if i > 0:
                    if target is not None and (np.random.uniform(0.0, 1.0)
                                               <= self.p_teacher_forcing):
                        target_idx = i * self.n_frames_per_step
                        decoder_input = target[:, target_idx - 1:target_idx]  # Teacher forcing
                    else:
                        decoder_input = last_projector_outputs[:, -1:]  # Auto-regression
                if self.pre_net is not None:
                    decoder_input, _ = self.pre_net(
                        decoder_input,
                        seq_lengths_input=seq_lengths_ones,
                        max_length_inputs=1)

            # TODO: add attention_context in _get_go_frame.
            attention_context, attention_weights, attention_state = self.attention.forward_incremental(
                i * self.n_frames_per_step,  # TODO: Has to become part of attention_context for fixed attention.
                input_,
                decoder_input,
                attention_state)
            all_attentions.append(attention_weights)

            decoder_input = torch.cat((attention_context, decoder_input), dim=-1)
            if self.model is not None:
                decoder_output, _ = self.model(
                    decoder_input,
                    seq_lengths_input=seq_lengths_ones,
                    max_length_inputs=1)
            else:
                decoder_output = decoder_input

            if self.decoder_output_name is not None:
                if self.decoder_output_name not in all_outputs_dict:
                    all_outputs_dict[self.decoder_output_name] = []
                all_outputs_dict[self.decoder_output_name].append(decoder_output)

            # TODO: Stop token predictor gets attention context as input as well.
            last_projector_outputs = list()
            for projector in self.projections:
                projector_output = projector.forward_module(
                    decoder_output, lengths, max_lengths)
                projector_output = self._parse_outputs(projector_output)
                if projector.output_names[0] in all_outputs_dict:
                    all_outputs_dict[projector.output_names[0]].append(projector_output)
                else:
                    all_outputs_dict[projector.output_names[0]] = [projector_output]
                if projector.is_autoregressive_input:
                    last_projector_outputs.append(projector_output)
            last_projector_outputs = torch.cat(last_projector_outputs, dim=-1)

        # Concat iterative outputs.
        for name, outputs in all_outputs_dict.items():
            all_outputs_dict[name] = torch.cat(outputs, dim=1)
        all_outputs_dict["attention"] = torch.cat(all_attentions, dim=1)

        # Add sequence lengths of projections.
        if lengths is not None \
                and self.teacher_forcing_input_names is not None \
                and self.teacher_forcing_input_names[0] in lengths:
            len_ = lengths[self.teacher_forcing_input_names[0]].clamp(
                max=max_steps)
            max_len = max_lengths[self.teacher_forcing_input_names[0]].clamp(
                min=max_steps)
        elif attention_matrix is not None \
                and self.attention.attention_ground_truth in lengths:
            len_ = lengths[self.attention.attention_ground_truth].clamp(
                max=max_steps)
            max_len = max_lengths[self.attention.attention_ground_truth].clamp(
                max=max_steps)
        else:
            # TODO: Compute based on StopTokenPrediction.
            len_ = max_len = torch.empty((B,), dtype=torch.long).fill_(max_steps)

        output_lengths_dict = {}
        output_max_lengths_dict = {}
        for projector in self.projections:
            for name in projector.output_names:
                output_lengths_dict[name] = len_
                output_max_lengths_dict[name] = max_len

        output_lengths_dict['attention'] = len_ // self.n_frames_per_step
        output_max_lengths_dict['attention'] = max_len // self.n_frames_per_step

        if self.decoder_output_name is not None:
            output_lengths_dict[self.decoder_output_name] = len_
            output_max_lengths_dict[self.decoder_output_name] = max_len

        return all_outputs_dict, output_lengths_dict, output_max_lengths_dict

    # def forward_fixed_attention_incremental(self, idx, step, input_,
    #                                         decoder_input, attention_matrix):
    #     # B = input_.shape[0]
    #     # T = input_.shape[1]
    #     # num_chunks = int(T / step)
    #     attention_weights = attention_matrix[:, idx:idx + step]
    #     attention_weights = attention_weights.mean(dim=1, keepdim=True)
    #     attention_context = torch.einsum("btp,bpc->btc",
    #                                      (attention_weights, input_))
    #     return attention_context, attention_weights
