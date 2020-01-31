#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#

import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pack_sequence, pad_sequence
import numpy as np
from idiaptts.misc.utils import parse_int_set


class RNNDyn(nn.Module):
    IDENTIFIER = "RNNDYN"

    # Supported nonlins. Nonlins are normally used in every layer, but only in the last layer in RNNs.
    # NOTE: FC and LIN are the same and meaning a fully-connected linear layer without non-linearity.
    nonlin_options = {'RELU': F.relu, 'TANH': torch.tanh, 'FC': None, 'LIN': None,
                      'SOFTMAX': F.softmax, 'LOGSOFTMAX': F.log_softmax, 'EXP': torch.exp}

    def __init__(self, dim_in, _, hparams):
        super().__init__()

        # Store parameters.
        self.use_gpu = hparams.use_gpu
        self.dim_in = dim_in
        self.dropout = hparams.dropout
        self.variable_sequence_length = hparams.variable_sequence_length_train
        self.batch_first = hparams.batch_first
        self.save_intermediate_outputs = hparams.save_intermediate_outputs if hasattr(hparams, "save_intermediate_outputs") and hparams.save_intermediate_outputs is not None else False

        # General dropout function.
        self.drop = nn.Dropout(hparams.dropout)

        # List of groups.
        self.layer_groups = nn.ModuleList()
        self.emb_groups = nn.ModuleList()

        # Translate model name to layer groups.
        assert(hparams.model_type is not None)
        assert(str(hparams.model_type).startswith(RNNDyn.IDENTIFIER))
        self.name_to_groups(hparams.model_type, hparams.hidden_init, hparams.train_hidden_init, hparams.f_get_emb_index if hasattr(hparams, "f_get_emb_index") else None)

        # Get output dimension from created model. Has to match with data.
        self.dim_out = self.layer_groups[-1].out_dim if len(self.layer_groups) > 0 else dim_in  # Consider special case where no network exists and input is just returned.

        # Select appropriate forward function.
        if hparams.variable_sequence_length_train:
            self.forward_var_seq_len = self.forward_var_seq_len_zero_pad
        else:
            self.forward_var_seq_len = self.forward_sample  # Only works without any zero padding.

    def forward(self, input, hidden, seq_lengths_input, max_length_inputs, target=None, seq_lengths_output=None):
        # Extra check to use simple forward for a single sample in batch.
        if len(seq_lengths_input) > 1:
            return self.forward_var_seq_len(input, hidden, seq_lengths_input, max_length_inputs, target, seq_lengths_output)
        else:
            return self.forward_sample(input, hidden, seq_lengths_input, max_length_inputs, target, seq_lengths_output)

    def set_gpu_flag(self, use_gpu):
        self.use_gpu = use_gpu

    def _setup_embeddings(self, in_dim, group_attr, f_get_emb_index):
        n_layers, embedding_dim = re.split('x', group_attr[0])
        n_layers = int(n_layers)
        embedding_dim = int(embedding_dim)
        emb = nn.Embedding(n_layers, embedding_dim)
        emb.affected_layer_indices = parse_int_set(group_attr[2].replace('(', '').replace(')', ''))
        if f_get_emb_index is None or len(f_get_emb_index) < len(self.emb_groups):
            raise ValueError("Embedding layer is defined but not enough f_get_emb_index functions are given in hparams.f_get_emb_index.")
        emb.f_get_emb_index = f_get_emb_index[len(self.emb_groups)]
        self.emb_groups.append(emb)
        in_dim -= int(np.prod(self.dim_in[1:]))  # Remove the embedding index from the input dimension.

        return in_dim

    def _gets_embedding(self, layer_idx):
        for emb in self.emb_groups:
            if -1 in emb.affected_layer_indices or layer_idx in emb.affected_layer_indices:
                return True
        return False

    def name_to_groups(self, name, hidden_init, train_hidden_init, f_get_emb_index):
        """
        Translate the name of a model to layer groups. Such a name always starts with "RNNDyn-".
        The groups are added to the nn.ModuleList self.layer_groups and are nn.ModuleLists itself.

        :param name:    Examples: RNNDyn-4_TANH_512-1_LSTM_63-1_RELU_512,
                                  RNNDyn-4_RELU_256-1_FC_10,
                                  RNNDyn-2_TANH_512-2_GRU_63-1_FC_2
                                  RNNDyn-1_LSTM_32-1_RELU_63
        :param hidden_init:       Float, value used to initialize the hidden states in RNNs
        :param train_hidden_init: Boolean, True if the initial hidden state value can be trained along with the network
        :return:        Nothing
        """

        str_layer_groups = re.split(r'-\s*(?![^()]*\))', name)  # Will give: type_identifier, (groups tuple)
        # if str_layer_groups[0] != self.IDENTIFIER:
        #     logging.warning("Name of network does not start with " + self.IDENTIFIER + "!")
        str_layer_groups = str_layer_groups[1:]  # Get rid of type_identifier.

        # Group string examples:
        # 4_TANH_512
        # 1_LSTM_63
        # 1_RELU_512

        in_dim = int(np.prod(self.dim_in))  # Use input size as first in_dim, is reduced by embeddings.
        out_dim = in_dim

        embeddings_done = False
        layer_idx = 0
        for group in str_layer_groups:
            # Split group string by underscore. The first three items always have to be set.
            # Layer who require more arguments can access group_attr[3:].
            group_attr = re.split('_', group)
            layer_type = group_attr[1]
            bidirectional = False
            if 'Bi' == layer_type[:2]:
                bidirectional = True
                layer_type = layer_type[2:]

            # First process embeddings.
            if layer_type == "EMB":
                if embeddings_done:
                    raise NotImplementedError("Please specify all embeddings layers before any other layers.")
                in_dim = self._setup_embeddings(in_dim, group_attr, f_get_emb_index)
                continue

            embeddings_done = True  # Set to true once a non-embedding layer is found.
                                    # Used to check that embedding layers are specified first.
            n_layers = int(group_attr[0])
            group_out_dim = int(group_attr[2])
            idx_sub_group = 0
            while idx_sub_group < n_layers:
                layer_group = nn.ModuleList()

                # Check if emb is involved.
                for emb in self.emb_groups:
                    if -1 in emb.affected_layer_indices or layer_idx in emb.affected_layer_indices:
                        in_dim += emb.embedding_dim
                    layer_group.in_dim = in_dim

                # Include all following layers which do not require an embedding.
                n_sub_group = 1
                for n in range(1, n_layers - idx_sub_group):
                    if self._gets_embedding(layer_idx + n):
                        break
                    n_sub_group += 1

                layer_group.n_layers = n_sub_group

                # Supported recurrent non-linearity.
                if layer_type in ['LSTM', 'GRU', 'RNNTANH', 'RNNRELU']:

                    layer_group.is_rnn = True
                    layer_group.nonlin = layer_type

                    out_dim = group_out_dim

                    if layer_type in ['LSTM', 'GRU']:
                        layer_group.append(getattr(nn, layer_type)(in_dim, out_dim, layer_group.n_layers,
                                                                   dropout=self.dropout if layer_group.n_layers > 1 else 0.0,
                                                                   bidirectional=bidirectional))
                    else:
                        rnn_opt = {'RNNTANH': 'tanh', 'RNNRELU': 'relu'}[layer_type]
                        layer_group.append(nn.RNN(in_dim, out_dim, layer_group.n_layers,
                                                  nonlinearity=rnn_opt,
                                                  dropout=self.dropout,
                                                  bidirectional=False))

                    num_directions = 2 if layer_group[0].bidirectional else 1
                    # group_out_dim = out_dim * num_directions
                    in_dim = out_dim * num_directions  # Next in_dim is the current outdim times the directions.

                    h0_init = torch.Tensor(layer_group.n_layers * num_directions, 1, out_dim).fill_(hidden_init)
                    c0_init = torch.Tensor(layer_group.n_layers * num_directions, 1, out_dim).fill_(hidden_init)

                    if train_hidden_init:
                        layer_group.register_parameter('h_0', nn.Parameter(h0_init))
                        layer_group.register_parameter('c_0', nn.Parameter(c0_init))
                    else:
                        layer_group.register_buffer('h_0', h0_init)
                        layer_group.register_buffer('c_0', c0_init)
                else:
                    layer_group.is_rnn = False
                    layer_group.nonlin = self.nonlin_options[layer_type]

                    # Options to pick different layers by name.
                    if True:
                        nn_layer = nn.Linear
                    else:
                        nn_layer = None

                    # Add requested number of layers.
                    for i in range(layer_group.n_layers):
                        out_dim = group_out_dim
                        layer_group.append(nn_layer(in_dim, out_dim))
                        in_dim = out_dim  # Next in_dim is the current out_dim.

                layer_group.out_dim = out_dim
                # Append to the modules layer groups list.
                self.layer_groups.append(layer_group)
                # Increment indices.
                layer_idx += layer_group.n_layers
                idx_sub_group += layer_group.n_layers

    def forward_sample(self, input, *_):
        """Forward one input through all layer groups, does not use the hidden parameter."""
        num_embs = len(self.emb_groups)
        input_embs = None
        if num_embs > 0:
            input_embs = input[:, :, -num_embs:]
            output = input[:, :, :-num_embs]
        else:
            output = input

        last_hidden = None
        layer_idx = 0
        for group in self.layer_groups:
            for emb_idx, emb in enumerate(self.emb_groups):
                if -1 in emb.affected_layer_indices or layer_idx in emb.affected_layer_indices:
                    output = torch.cat((output, emb(input_embs[:, :, emb_idx].long())), dim=2)

            if group.is_rnn:
                assert hasattr(group, "hidden"), "If group.hidden is not set here, init_hidden was not called."
                output, group.hidden = group[0](output, group.hidden)
                if group.hidden is not None:  # Keep last not None hidden state.
                    last_hidden = group.hidden

                output = self.drop(output)  # Dropout is by default not applied to last rnn layer.
            else:
                if group.nonlin is None:
                    for layer in group:
                        output = self.drop(layer(output))
                else:
                    for layer in group:
                        if group.nonlin is F.softmax:
                            output = self.drop(group.nonlin(layer(output), dim=2))
                        else:
                            output = self.drop(group.nonlin(layer(output)))

            layer_idx += group.n_layers if hasattr(group, "n_layers") else 1  # Backwards compatibility.
            # Only save the output of each group.
            if hasattr(self, "save_intermediate_outputs") and self.save_intermediate_outputs:
                group.output = output

        return output, last_hidden

    def forward_var_seq_len_zero_pad(self, input, hidden, seq_lengths_input, max_length_inputs, *_):
        """Forward one input through all layer groups, does not use the hidden parameter."""
        num_embs = len(self.emb_groups)
        input_embs = None
        if num_embs > 0:
            input_embs = input[:, :, -num_embs:]
            output = input[:, :, :-num_embs]
        else:
            output = input

        last_hidden = None
        layer_idx = 0
        for group in self.layer_groups:
            for emb_idx, emb in enumerate(self.emb_groups):
                if -1 in emb.affected_layer_indices or layer_idx in emb.affected_layer_indices:
                    output = torch.cat((output, emb(input_embs[:, :, emb_idx].long())), dim=2)

            if group.is_rnn:
                # Pack the sequence for RNN.
                total_length = output.size(1)  # Get max sequence length, required to use pad_packed_sequence with data parallel. # TODO: Requires testing.
                output = pack_padded_sequence(output, seq_lengths_input)  # TODO: Accept batch first

                output, group.hidden = group[0](output, group.hidden)  # If group.hidden is not set here, init_hidden was not called.
                if group.hidden is not None:  # Keep last not None hidden state.
                    last_hidden = group.hidden

                # Reverse pack_padded_sequence operation to work with dropout.
                output, _ = pad_packed_sequence(output, total_length=max_length_inputs)

                output = self.drop(output)  # Dropout is by default not applied to last rnn layer.
            else:
                for layer in group:
                    # Pass through layer.
                    output = layer(output)
                    # Apply non-linearity.
                    if group.nonlin is F.softmax:
                        output = group.nonlin(output, dim=2)
                    elif group.nonlin is not None:
                        output = group.nonlin(output)
                    # Apply dropout.
                    output = self.drop(output)

            layer_idx += group.n_layers if hasattr(group, "n_layers") else 1  # Backwards compatibility.
            # Only save the output of each group.
            if hasattr(self, "save_intermediate_outputs") and self.save_intermediate_outputs:
                group.output = output

        return output, last_hidden

    def forward_var_seq_len_concat(self, input, hidden, seq_lengths_input, max_length_inputs, target=None, seq_lengths_output=None):
        """Forward one input through all layer groups, does not use the hidden parameter."""
        raise NotImplementedError("This function is no longer supported.")

    def init_hidden(self, batch_size=1):
        for group in self.layer_groups:
            if group.is_rnn:
                group.hidden = self.group_init_hidden(group, batch_size)
        return None

    def group_init_hidden(self, group, batch_size=1):
        """Get init hidden state for a group."""
        hidden_size = list(group.h_0.size())
        hidden_size[1] = batch_size

        h_0 = group.h_0.expand(hidden_size).contiguous()

        if group.nonlin in ['LSTM', 'LSTMP', 'BiLSTM', 'BiLSTMP']:
            c_0 = group.c_0.expand(hidden_size).contiguous()
            return h_0, c_0
        else:
            return h_0

    def __getitem__(self, item):
        """Zero-based indexing of layers."""

        layer_idx = 0
        for group in self.layer_groups:
            if layer_idx + group.n_layers > item:
                if group.is_rnn:
                    return group[0]  # RNN layers contain multiple layers in the same object, so return that object.
                else:
                    return group[item - layer_idx]
            else:
                layer_idx += group.n_layers
        raise KeyError("The model contains only {} layers.".format(layer_idx))

    def get_embeddings(self):
        return self.emb_groups

    def get_group_out_dim(self, group_idx):
        group = self.layer_groups[group_idx]
        return group.out_dim * (2 if group.is_rnn else 1)

    def get_intermediate_output(self, group_idx=-1):
        assert self.save_intermediate_outputs,\
            "Cannot access intermediate output because hparams.save_intermediate_outputs is False."
        return self.layer_groups[group_idx].output


class MerlinAcoustic(RNNDyn):
    IDENTIFIER = "MerlinAcoustic"

    def __init__(self, dim_in, dim_out, hparams):
        hparams.model_type = super().IDENTIFIER + "-6_TANH_512-1_FC_" + str(np.prod(dim_out))
        super().__init__(dim_in, dim_out, hparams)
        hparams.model_type = self.IDENTIFIER


class Interspeech18baseline(RNNDyn):
    IDENTIFIER = "Interspeech18baseline"

    def __init__(self, dim_in, dim_out, hparams):
        hparams.model_type = super().IDENTIFIER + "-2_RELU_1024-3_BiGRU_512-1_FC_" + str(np.prod(dim_out))
        super().__init__(dim_in, dim_out, hparams)
        hparams.model_type = self.IDENTIFIER


class Icassp19baseline(RNNDyn):
    IDENTIFIER = "Icassp19baseline"

    def __init__(self, dim_in, dim_out, hparams):
        hparams.model_type = super().IDENTIFIER + "-2_RELU_1024-3_BiLSTM_512-1_FC_" + str(np.prod(dim_out))
        super().__init__(dim_in, dim_out, hparams)
        hparams.model_type = self.IDENTIFIER


class BaselineRNN_Yamagishi(RNNDyn):
    IDENTIFIER = "BaselineRNN_Yamagishi"

    def __init__(self, dim_in, dim_out, hparams):
        hparams.model_type = super().IDENTIFIER + "-2_RELU_1024-3_BiGRU_512-1_FC_" + str(np.prod(dim_out))
        super().__init__(dim_in, dim_out, hparams)
        hparams.model_type = self.IDENTIFIER
