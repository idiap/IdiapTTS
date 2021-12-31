#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#

import re
import copy
import logging

import torch
import torch.nn as nn
import numpy as np
import jsonpickle

from . import IDENTIFIER
from .Config import Config
from .FFWrapper import FFWrapper
from .TransposingWrapper import TransposingWrapper
from .CNNWrapper import CNNWrapper
from .RNNWrapper import RNNWrapper
from idiaptts.misc.utils import parse_int_set


class RNNDyn(nn.ModuleList):

    def __init__(self, config):

        super(RNNDyn, self).__init__()
        # if len(args) > 1:
        #     config = self._get_config_from_legacy_string(*args)
        # else:
        #     config = args[0]
        self.config = copy.deepcopy(config)

        self._setup_embeddings(config.emb_configs)
        self._setup_layers(config.batch_first, config.in_dim,
                           config.layer_configs)

    def _setup_embeddings(self, emb_configs):
        self.emb_groups = nn.ModuleDict()
        if emb_configs is not None:
            for emb_config in emb_configs:
                emb = nn.Embedding(emb_config.num_embedding,
                                   emb_config.embedding_dim)
                emb.affected_layer_group_indices = \
                    emb_config.affected_layer_group_indices
                emb.name = emb_config.name
                self.emb_groups[emb_config.name] = emb

    def _setup_layers(self, batch_first, in_dim, layer_configs):
        in_dim = in_dim
        self.layer_groups = []
        last_was_transposed = False
        last_was_packed = False
        last_layer = None
        for group_idx, layer_config in enumerate(layer_configs):
            in_dim += self._get_embeddings_dim(group_idx)
            if layer_config.needs_packing:
                layer = RNNWrapper(in_dim, layer_config, batch_first,
                                   enforce_sorted=False)
                if last_was_packed and not self._has_emb_inputs(group_idx):
                    last_layer.unpack = False
                    layer.pack = False

            elif layer_config.needs_transposing:
                if layer_config.type == "Conv1d":
                    layer = CNNWrapper(in_dim, layer_config, batch_first)
                else:
                    # Batch norm layers need transposing too.
                    layer = TransposingWrapper(in_dim, layer_config,
                                               batch_first)
                if last_was_transposed and not self._has_emb_inputs(group_idx):
                    last_layer.untranspose = False
                    layer.transpose = False
            else:
                layer = FFWrapper(in_dim, layer_config)

            in_dim = layer.out_dim
            last_was_transposed = layer_config.needs_transposing
            last_was_packed = layer_config.needs_packing
            last_layer = layer
            self.append(layer)
            self.layer_groups.append(layer)

    def forward(self, input_, *emb_inputs, **kwargs):
        num_emb_groups = len(self.emb_groups)
        num_input_embs = len(emb_inputs)
        if num_emb_groups > 0:
            if num_input_embs > 0:
                if num_input_embs != num_emb_groups:
                    raise ValueError("Given number of embedding inputs ({}) "
                                     "does not match number of embedding "
                                     "groups ({}) in model.".format(
                                         num_input_embs, num_emb_groups))
            else:
                logging.warning("Do not concatenate embedding input indices "
                                "to the input. Use separate arguments for "
                                "input and each embedding.",
                                DeprecationWarning)
                emb_inputs = input_[:, :, -num_emb_groups:]
                emb_inputs = torch.split(emb_inputs,
                                         split_size_or_sections=num_emb_groups,
                                         dim=-1)
                input_ = input_[:, :, :-num_emb_groups]

            embeddings = self._compute_embeddings(*emb_inputs)
        else:
            emb_inputs = None
            embeddings = {}

        last_hidden = None
        for group_idx, module in enumerate(self.layer_groups):
            input_ = self._add_embeddings(group_idx, input_, embeddings)
            input_, kwargs = module(input_, **kwargs)
            # Do not pass the hidden state of one layer to the following
            # RNN layers.
            last_hidden = kwargs.pop("hidden", last_hidden)
        kwargs["hidden"] = last_hidden
        return input_, kwargs

    def _compute_embeddings(self, *emb_inputs):
        embeddings = {}
        for idx, emb in enumerate(self.emb_groups.values()):
            embeddings[emb.name] = emb(emb_inputs[idx][:, :, 0].long())
        return embeddings

    def _add_embeddings(self, group_idx, input_, embeddings):
        for emb in self.emb_groups.values():
            if -1 in emb.affected_layer_group_indices \
                    or group_idx in emb.affected_layer_group_indices:
                input_ = torch.cat((input_, embeddings[emb.name]), dim=2)
        return input_

    def _get_embeddings_dim(self, group_idx):
        dim = 0
        for emb in self.emb_groups.values():
            if -1 in emb.affected_layer_group_indices \
                    or group_idx in emb.affected_layer_group_indices:
                dim += emb.embedding_dim
        return dim

    def _has_emb_inputs(self, group_idx):
        return self._get_embeddings_dim(group_idx) != 0

    def set_gpu_flag(self, use_gpu):
        self.use_gpu = use_gpu

    @staticmethod
    def _get_config_from_legacy_string(in_dim, _, hparams):
        """
        Translate the name of a model to layer groups. Such a name
        always starts with "RNNDyn-". The groups are added to the
        nn.ModuleList self.layer_groups and are nn.ModuleLists itself.

        :param name: Examples: RNNDyn-4_TANH_512-1_LSTM_63-1_RELU_512,
                               RNNDyn-4_RELU_256-1_FC_10,
                               RNNDyn-2_TANH_512-2_GRU_63-1_FC_2
                               RNNDyn-1_LSTM_32-1_RELU_63
        :param in_dim: Dimension of input tensor.
        :param _: Used to be out_dim but is unused here, it remains to
            support legacy calls.
        :param hparams: Hyper-parameter container.
        :return: New style config.
        """
        logging.warning("Convert to new style configs for RNNDyn.",
                        DeprecationWarning)

        # Supported nonlins. Nonlins are normally used in every layer,
        # but only in the last layer in RNNs. NOTE: FC and LIN are the
        # same and meaning a fully-connected linear layer without
        # non-linearity.
        nonlin_options = {'RELU': "ReLU", 'TANH': "Tanh"}

        name = hparams.model_type

        if hasattr(hparams, 'f_get_emb_index'):
            f_get_emb_index = hparams.f_get_emb_index
        else:
            f_get_emb_index = None
        batch_first = hparams.batch_first
        dropout = hparams.dropout

        # The split will give: type_identifier, (groups tuple)
        str_layer_groups = re.split(r'-\s*(?![^()]*\))', name)
        str_layer_groups = str_layer_groups[1:]  # Get rid of type_identifier.

        # Group string examples:
        # 4_TANH_512
        # 1_LSTM_63
        # 1_RELU_512

        # Use input size as first in_dim, is reduced by embeddings.
        in_dim_without_embs = int(np.prod(in_dim))
        embeddings_done = False
        embedding_configs = []
        layer_configs = []
        if len(str_layer_groups) == 0:
            raise ValueError("Empty {} configuration: {}".format(IDENTIFIER,
                                                                 name))

        for group in str_layer_groups:
            # Split group string by underscore. The first three items
            # always have to be set. Layer who require more arguments
            # can access group_attr[3:].
            group_attr = re.split('_', group)
            layer_type = group_attr[1]
            bidirectional = False
            if 'Bi' == layer_type[:2]:
                bidirectional = True
                layer_type = layer_type[2:]

            # First process embeddings.
            if layer_type == "EMB":
                if embeddings_done:
                    raise NotImplementedError(
                        "Please specify all embeddings layers before any "
                        "other layers.")

                num_embeddings, embedding_dim = re.split('x', group_attr[0])
                num_embeddings = int(num_embeddings)
                embedding_dim = int(embedding_dim)
                affected_layer_group_indices = parse_int_set(
                    group_attr[2].replace('(', '').replace(')', ''))
                if f_get_emb_index is None \
                        or len(f_get_emb_index) < len(embedding_configs):
                    raise ValueError("Embedding layer is defined but not "
                                     "enough f_get_emb_index functions are"
                                     "given in hparams.f_get_emb_index.")
                embedding_configs.append(Config.EmbeddingConfig(
                    embedding_dim, str(len(embedding_configs)), num_embeddings,
                    affected_layer_group_indices))
                # Remove the embedding index from the input dimension.
                in_dim_without_embs -= int(np.prod(in_dim[1:]))
                continue

            # Set embeddings_done to true once a non-embedding layer is
            # found. Used to check that # embedding layers are specified
            # first.
            embeddings_done = True

            n_layers = int(group_attr[0])
            group_out_dim = int(group_attr[2])
            norm_type = None

            # Handle normalisation here.
            for available_norm_type in ["BatchNorm1d", ]:
                if layer_type.startswith(available_norm_type):
                    norm_type = available_norm_type
                    if n_layers > 1:
                        raise NotImplementedError(
                            "{} is not supported for more than one layer "
                            "(but requested {}).".format(norm_type, n_layers)
                            + " Please specify the layers separately.")
                    layer_type = layer_type[len(available_norm_type):]
                    break  # There should be only one type of normalisation, I guess?

            if layer_type.upper() in ['SOFTMAX', 'LOGSOFTMAX', 'EXP']:
                raise NotImplementedError("{} is not supported in legacy"
                                          "string. Use new style configs"
                                          " instead.".format(layer_type))
            if layer_type.upper() in nonlin_options:
                nonlin = nonlin_options[layer_type.upper()]
            else:
                nonlin = None

            # Supported recurrent non-linearity.
            if layer_type in ['LSTM', 'GRU', 'RNNTANH', 'RNNRELU']:
                if 'RNN' in layer_type:
                    nonlin = {'RNNTANH': 'tanh', 'RNNRELU': 'relu'}[layer_type]
                    layer_type = 'RNN'
                layer_config = Config.LayerConfig(
                    layer_type=layer_type,
                    out_dim=group_out_dim,
                    num_layers=n_layers,
                    nonlin=None,
                    dropout=dropout if n_layers > 1 else 0.0,
                    bidirectional=bidirectional)
                layer_config.nonlin = nonlin

            elif layer_type.startswith("Conv1d"):  # Could generalise to 2d and 3d.
                # Example for convolutional network:
                #    3_BatchNorm1dConv1dRELU_512_5
                # 3 layers of 1D convolutions with 512 neurons and a
                # kernel of (5,) followed by ReLU and batch norm.
                if len(group_attr) < 4:
                    raise NotImplementedError(
                        "Kernel size has to be given in {} as ".format(
                            layer_type) +
                        "<num layer>_<layer type>_<out dim>_<kernel size(s)>.")
                kernel = tuple(map(int, group_attr[3].split('x')))
                stride = 1
                # Designed so that sequence length remains the same.
                # Usefull for speech.
                padding = int((kernel[0] - 1) / 2)
                dilation = 1
                conv_groups = 1
                for param in group_attr[4:]:
                    if param[0] == "s":
                        stride = tuple(map(int, param[1:].split('x')))
                    elif param[0] == "p":
                        padding = tuple(map(int, param[1:].split('x')))
                    elif param[0] == "d":
                        dilation = tuple(map(int, param[1:].split('x')))
                    elif param[0] == "g":
                        conv_groups = int(param[1:])
                    else:
                        raise NotImplementedError(
                            "Unknown param type {} in {}".format(
                                param, str_layer_groups))

                layer_config = Config.LayerConfig(layer_type=layer_type,
                                                  out_dim=group_out_dim,
                                                  num_layers=n_layers,
                                                  kernel_size=kernel,
                                                  stride=stride,
                                                  padding=padding,
                                                  dilation=dilation,
                                                  groups=conv_groups)
            elif layer_type.startswith("Emb"):
                embedding_dim, num_embeddings = group_attr[2:]
                layer_config = Config.LayerConfig(
                    layer_type="Embedding",
                    out_dim=embedding_dim,
                    num_embeddings=int(num_embeddings))
            elif layer_type.startswith("Pool"):
                if layer_type == "PoolLast":
                    layer_config = Config.LayerConfig(
                        layer_type="SelectLastPooling",
                        batch_first=batch_first)
                else:
                    raise NotImplementedError()
            elif "VAE" in layer_type:
                if layer_type == "VAE" or layer_type == "VanillaVAE":
                    layer_config = Config.LayerConfig(layer_type="VanillaVAE",
                                                      out_dim=group_out_dim)
                else:
                    raise NotImplementedError()
            else:
                layer_config = Config.LayerConfig(layer_type='Linear',
                                                  out_dim=group_out_dim,
                                                  num_layers=n_layers,
                                                  nonlin=nonlin,
                                                  dropout=dropout)

            layer_configs.append(layer_config)

            if norm_type is not None:
                norm_layer_config = Config.LayerConfig(layer_type=norm_type,
                                                       out_dim=group_out_dim)
                layer_configs.append(norm_layer_config)

        return Config(in_dim=in_dim_without_embs,
                      batch_first=batch_first,
                      layer_configs=layer_configs,
                      emb_configs=embedding_configs)

    def init_hidden(self, batch_size=1):
        for module in self.layer_groups:
            module.init_hidden(batch_size)

    def __getitem__(self, item):
        """Zero-based indexing of layers."""
        return self.layer_groups[item]

    def get_embeddings(self):
        return self.emb_groups

    def get_group_out_dim(self, group_idx):
        group = self.layer_groups[group_idx]
        return group.out_dim

    def get_config_as_json(self):
        return jsonpickle.encode(self.config, indent=4)


# TODO: Convert the following classes to new style configs.
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
