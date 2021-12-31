#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#

import copy
from typing import Union, List, Tuple


class Config:

    class LayerConfig:
        layer_type_map = {
            "FC": "Linear",
            "linear": "Linear",
            "LIN": "Linear"
        }

        def __init__(self, layer_type: str, out_dim: int = None,
                     needs_in_dim: bool = None, num_layers: int = 1,
                     nonlin: str = None, dropout: float = 0.0,
                     **kwargs) -> None:
            self.type = self._map_layer_type(layer_type)
            self.out_dim = out_dim
            self.num_layers = num_layers
            self.dropout = dropout
            self.kwargs = kwargs

            self.nonlin = nonlin
            # Handle upper/lower case versions of common non-linearities.
            if nonlin is not None:
                nonlin_lower = nonlin.lower()
                if nonlin_lower == "relu":
                    self.nonlin = "ReLU"
                elif nonlin_lower == "tanh":
                    self.nonlin = "Tanh"

            if needs_in_dim is not None:
                self.needs_in_dim = needs_in_dim
            # Add layers which require the input dimension here.
            elif self.type in ["Linear"] \
                or ('Norm' in layer_type
                    and layer_type not in ['GroupNorm', 'LayerNorm']):
                self.needs_in_dim = True
            else:
                self.needs_in_dim = False
            # Add layers which require a packed sequence here.
            self.needs_packing = layer_type in ['LSTM', 'GRU', 'RNN']
            # Add layers which require the feature dimension as second
            # dimension here.
            self.needs_transposing = 'Conv' in layer_type \
                or 'BatchNorm' in layer_type

        def _map_layer_type(self, type_):
            if type_ in self.layer_type_map:
                return self.layer_type_map[type_]
            else:
                return type_

        def __repr__(self):
            representation = "{}x {} {}".format(
                self.num_layers,
                self.out_dim if self.out_dim is not None else "",
                self.type)

            if self.nonlin is not None:
                representation += " with {}".format(self.nonlin)

            if self.dropout is not None and self.dropout != 0.0:
                representation += ", dropout: {}".format(self.dropout)

            if len(self.kwargs) > 0:
                representation += ", " + ", ".join(
                    ["{}: {}".format(key, value)
                     for key, value in self.kwargs.items()])

            return representation

    class EmbeddingConfig:
        def __init__(self, embedding_dim, name: str, num_embedding,
                     affected_layer_group_indices: Union[int, Tuple[int, ...],
                                                         List[int]] = -1):
            """

            :param embedding_dim: Dimension of the embedding vector.
            :param name: Used to retrieve the embedding index from the
                data dictionary.
            :param num_embedding: Maximum accepted embedding index.
            :param affected_layer_group_indices: Tuple of integers
                specifying which layer *groups* get the embedding as
                additional input. NOTE: When using multiple layers of
                the same type only the first layer of the group will
                receive the embedding as additional input. If you want
                it as input to all layers, specify them in separate
                layer configs.
            """
            self.embedding_dim = embedding_dim
            self.name = name
            self.num_embedding = num_embedding
            if affected_layer_group_indices is None \
                or type(affected_layer_group_indices) in [tuple, list, set]:
                self.affected_layer_group_indices = affected_layer_group_indices
            else:
                self.affected_layer_group_indices = (affected_layer_group_indices,)

        def __repr__(self):
            return "{}: {} inputs, {} embedding dim, affected groups: ({})" \
                .format(self.name, self.num_embeddings, self.embedding_dim,
                        ", ".join(self.affected_layer_group_indices))

    def __init__(self,
                 config_str: str = None,
                 in_dim: int = 0,
                 hparams=None,
                 batch_first: bool = True,
                 layer_configs: List[LayerConfig] = None,
                 emb_configs: List[EmbeddingConfig] = None):

        assert in_dim > 0
        self.in_dim = in_dim
        if config_str is not None:
            self.config_str = config_str
            self.hparams = copy.deepcopy(hparams)
            self.hparams.model_type = config_str

        self.batch_first = batch_first
        self.layer_configs = layer_configs
        self.emb_configs = emb_configs

    def create_model(self):
        from .RNNDyn import RNNDyn  # Import here to break cyclic dependency.
        # Legacy support
        if self.layer_configs is None:
            return RNNDyn(self.in_dim, None, self.hparams)
        else:
            return RNNDyn(self)
