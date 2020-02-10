#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#


import unittest

import os
import shutil
from functools import partial

import torch
import numpy

from idiaptts.src.model_trainers.vtln.VTLNSpeakerAdaptionModelTrainer import VTLNSpeakerAdaptionModelTrainer
from idiaptts.misc.utils import makedirs_safe
from idiaptts.src.neural_networks.pytorch.models.AllPassWarpModel import AllPassWarpModel
from idiaptts.src.neural_networks.EmbeddingConfig import EmbeddingConfig


class TestAllPassWarpModel(unittest.TestCase):

    def test_init_single_embedding(self):
        hparams = VTLNSpeakerAdaptionModelTrainer.create_hparams()
        hparams.num_coded_sps = 20
        hparams.add_deltas = True
        hparams.load_pre_net_from_checkpoint = False
        hparams.add_hparam("embeddings", [(EmbeddingConfig(lambda x: 0, 2, 24), True, True), ])

        in_dim = 150
        out_dim = (hparams.num_coded_sps + 1 + hparams.num_bap) * (3 if hparams.add_deltas else 1) + 1  # +1 for VUV.

        hparams.del_hparam("alpha_ranges")  # Delete default value.
        with self.assertRaises(AssertionError) as cm:
            AllPassWarpModel(in_dim, out_dim, hparams)
        self.assertEqual("Embedding 0 should be passed to all warping layers but no warping layers exist.",
                         str(cm.exception))

        hparams.add_hparam("alpha_ranges", [0.2, ])
        with self.assertRaises(AssertionError) as cm:
            AllPassWarpModel(in_dim, out_dim, hparams)
        self.assertEqual("Embedding 0 should be passed to pre-net but no pre-net exists.", str(cm.exception))

        hparams.pre_net_model_type = "RNNDYN-1_RELU_32-1_BiGRU_16-1_FC_{}".format(out_dim)
        with self.assertRaises(AssertionError) as cm:
            AllPassWarpModel(in_dim, out_dim, hparams)
        self.assertEqual("Embedding 0 should be passed to pre-net but pre-net expects only 0 embeddings.",
                         str(cm.exception))

        hparams.pre_net_model_type = "RNNDYN-2x24_EMB_(-1)-1_RELU_32-1_BiGRU_16-1_FC_{}".format(out_dim)
        model = AllPassWarpModel((in_dim,), (out_dim,), hparams)

        self.assertEqual(torch.Size((1, 2 * 16 + 24)), model.alpha_layers[0].weight.shape)

    def test_init_double_embedding(self):
        hparams = VTLNSpeakerAdaptionModelTrainer.create_hparams()
        hparams.num_coded_sps = 20
        hparams.add_deltas = True
        hparams.load_pre_net_from_checkpoint = False
        hparams.add_hparam("embeddings", [(EmbeddingConfig(lambda x: 0, 2, 24), True, [False, True]),
                                          (EmbeddingConfig(lambda x: 1, 3, 15), False, True)])
        hparams.alpha_ranges.append(0.1)
        in_dim = 150
        out_dim = (hparams.num_coded_sps + 1 + hparams.num_bap) * (3 if hparams.add_deltas else 1) + 1  # +1 for VUV.
        hparams.pre_net_model_type = "RNNDYN-2x24_EMB_(-1)-1_RELU_32-1_BiGRU_16-1_FC_{}".format(out_dim)

        model = AllPassWarpModel((in_dim,), (out_dim,), hparams)
        self.assertEqual(torch.Size((32, in_dim - 1 + 24)), model.pre_net.layer_groups[0][0].weight.shape)
        self.assertEqual(torch.Size((1, 2 * 16 + 15)), model.alpha_layers[0].weight.shape)
        self.assertEqual(torch.Size((1, 2 * 16 + 24 + 15)), model.alpha_layers[1].weight.shape)

    def test_init_triple_embedding(self):
        hparams = VTLNSpeakerAdaptionModelTrainer.create_hparams()
        hparams.num_coded_sps = 20
        hparams.add_deltas = True
        hparams.load_pre_net_from_checkpoint = False
        hparams.add_hparam("embeddings", [(EmbeddingConfig(lambda x: 0, 2, 24), True, [False, True]),
                                          (EmbeddingConfig(lambda x: 1, 3, 15), False, True),
                                          (EmbeddingConfig(lambda x: 1, 3, 100), True, False)])
        hparams.alpha_ranges.append(0.1)
        in_dim = 150
        out_dim = (hparams.num_coded_sps + 1 + hparams.num_bap) * (3 if hparams.add_deltas else 1) + 1  # +1 for VUV.
        hparams.pre_net_model_type = "RNNDYN-2x24_EMB_(-1)-3x100_EMB_(-1)-1_RELU_32-1_BiGRU_16-1_FC_{}".format(out_dim)

        model = AllPassWarpModel((in_dim,), (out_dim,), hparams)
        self.assertEqual(torch.Size((32, in_dim - 2 + 24 + 100)), model.pre_net.layer_groups[0][0].weight.shape)
        self.assertEqual(torch.Size((1, 2 * 16 + 15)), model.alpha_layers[0].weight.shape)
        self.assertEqual(torch.Size((1, 2 * 16 + 24 + 15)), model.alpha_layers[1].weight.shape)

    def test_train_single_embedding(self):
        hparams = VTLNSpeakerAdaptionModelTrainer.create_hparams()
        hparams.num_coded_sps = 20
        hparams.add_deltas = True
        hparams.load_pre_net_from_checkpoint = False
        hparams.add_hparam("embeddings", [(EmbeddingConfig(lambda x: 0, 2, 24), True, [False, True]), ])
        hparams.alpha_ranges.append(0.1)
        in_dim = 150
        out_dim = (hparams.num_coded_sps + 1 + hparams.num_bap) * (3 if hparams.add_deltas else 1) + 1  # +1 for VUV.
        hparams.pre_net_model_type = "RNNDYN-2x24_EMB_(-1)-1_RELU_32-1_BiGRU_16-1_FC_{}".format(out_dim)

        torch.manual_seed(42)
        model = AllPassWarpModel((in_dim,), (out_dim,), hparams)

        batch_size = 2
        T = (250, 200)
        input = torch.rand((T[0], batch_size, in_dim), requires_grad=True)
        model.init_hidden(batch_size)
        output, hidden = model(input, None, T, max(T))

        self.assertEqual(torch.Size((T[0], batch_size, out_dim)),
                         output.shape)

        output.sum().backward()
        expected_input_grad = torch.tensor(-154.8480)
        input_grad = input.grad.sum()
        self.assertTrue(expected_input_grad.isclose(input_grad),
                        "Cummulated input gradient is {} but expected {}.".format(input_grad, expected_input_grad))
