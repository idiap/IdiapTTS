#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#


class EmbeddingConfig(object):

    def __init__(self, f_get_emb_index, num_embeddings, embedding_dim, name=None, **args):
        assert callable(f_get_emb_index), "f_get_emb_index must be callable."
        self.f_get_emb_index = f_get_emb_index
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.name = name
        self.args = args

    def __repr__(self):
        if self.name is None:
            output = ""
        else:
            output = "{}: ".format(self.name)
        output += "{}x{}".format(self.num_embeddings, self.embedding_dim)
        if len(self.args.keys()) > 0:
            output += "with " + " ".join(map(str, **self.args))
        return output
