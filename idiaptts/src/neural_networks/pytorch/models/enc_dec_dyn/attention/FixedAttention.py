#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#

from idiaptts.src.neural_networks.pytorch.models.enc_dec_dyn.attention.Attention import Attention
import torch


class FixedAttention(Attention):

    class Config:
        def __init__(self, ground_truth_feature_name, n_frames_per_step) -> None:
            self.ground_truth_feature_name = ground_truth_feature_name
            self.n_frames_per_step = n_frames_per_step

        def create_model(self):
            return FixedAttention(self)

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.attention_ground_truth = config.ground_truth_feature_name
        self.n_frames_per_step = config.n_frames_per_step

    def allows_batched_forward(self):
        return True

    def forward_batched(self, encoder_input, attention_matrix):
        B = attention_matrix.shape[0]
        T = attention_matrix.shape[1]
        num_chunks = int(T / self.n_frames_per_step)
        # attention_matrix = attention_matrix[:, ::self.n_frames_per_step]  # Skip frames.
        attention_matrix = attention_matrix.view(
            B, num_chunks, self.n_frames_per_step, -1).mean(dim=2)
        attention_context = torch.bmm(attention_matrix, encoder_input)
        return attention_context, attention_matrix

    def forward_incremental(self, idx, encoder_input, decoder_input, attention_matrix):
        # B = input_.shape[0]
        # T = input_.shape[1]
        # num_chunks = int(T / step)
        attention_weights = attention_matrix[:, idx:idx + self.n_frames_per_step]
        attention_weights = attention_weights.mean(dim=1, keepdim=True)
        attention_context = torch.einsum("btp,bpc->btc",
                                         (attention_weights, encoder_input))
        return attention_context, attention_weights, attention_matrix
