#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#


import unittest

import os
import shutil
import torch
import numpy

from idiaptts.misc.utils import makedirs_safe
from idiaptts.src.model_trainers.ModelTrainer import ModelTrainer
from idiaptts.src.neural_networks.pytorch.ModelFactory import ModelFactory


class TestRNNDyn(unittest.TestCase):

    out_dir = None

    @classmethod
    def setUpClass(cls):
        cls.out_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), type(cls()).__name__)
        makedirs_safe(cls.out_dir)  # Create class name directory.

    @classmethod
    def tearDownClass(cls):
        os.rmdir(cls.out_dir)  # Remove class name directory, should be empty.

    def test_get_item(self):
        hparams = ModelTrainer.create_hparams()
        num_emb = 3
        emb_dim = 12
        in_dim = 42
        out_dim = 12
        hparams.add_hparam("f_get_emb_index", [lambda x: 0])
        hparams.model_type = "RNNDYN-{}x{}_EMB_(0, 3, 5, 7)-5_RELU_128-3_BiLSTM_32-1_FC_12".format(num_emb, emb_dim)
        model = ModelFactory.create(hparams.model_type, (in_dim,), out_dim, hparams)

        self.assertEqual(model.layer_groups[0][1], model[1])
        self.assertEqual(model.layer_groups[1][0], model[3])
        self.assertEqual(model.layer_groups[2][0], model[6])

    def test_embeddings(self):
        hparams = ModelTrainer.create_hparams()
        num_emb = 3
        emb_dim = 12
        in_dim = 42  # Contains the embedding index.
        out_dim = 12
        hparams.variable_sequence_length_train = True
        hparams.add_hparam("f_get_emb_index", [lambda x: 0])
        hparams.model_type = "RNNDYN-{}x{}_EMB_(0, 3, 5, 7)-5_RELU_128-3_BiLSTM_32-1_FC_12".format(num_emb, emb_dim)
        # hparams.model_type = "RNNDYN-{}x{}_EMB_(-1)-5_RELU_128-2_BiLSTM_32-1_FC_12".format(num_emb, emb_dim)
        model = ModelFactory.create(hparams.model_type, (in_dim,), out_dim, hparams)

        self.assertEqual(1, len(model.emb_groups))
        self.assertEqual(torch.Size([num_emb, emb_dim]), model.emb_groups[0].weight.shape)
        self.assertEqual(torch.Size([128, in_dim - 1 + emb_dim]), model[0].weight.shape)
        self.assertEqual(torch.Size([128, 128]), model[2].weight.shape)
        self.assertEqual(torch.Size([128, 128 + emb_dim]), model[3].weight.shape)

        self.assertEqual(torch.Size([32 * 4, 128 + emb_dim]), model[5].weight_ih_l0.shape)
        self.assertEqual(torch.Size([32 * 4, 32 * 2 + emb_dim]), model[7].weight_ih_l0_reverse.shape)

        seq_length = torch.tensor((100, 75), dtype=torch.long)
        batch_size = 2
        test_input = torch.ones([seq_length[0], batch_size, in_dim])
        model.init_hidden(batch_size)
        output = model(test_input, None, seq_length, seq_length[0])
        self.assertEqual(torch.Size([seq_length[0], batch_size, out_dim]), output[0].shape)

        seq_length = torch.tensor((100,), dtype=torch.long)
        batch_size = 1
        test_input = torch.ones([seq_length[0], batch_size, in_dim])
        model.init_hidden(batch_size)
        output = model(test_input, None, seq_length, seq_length[0])
        self.assertEqual(torch.Size([seq_length[0], batch_size, out_dim]), output[0].shape)

    def test_embeddings_everywhere(self):
        hparams = ModelTrainer.create_hparams()
        num_emb = 3
        emb_dim = 12
        in_dim = 42
        out_dim = 12
        hparams.add_hparam("f_get_emb_index", [lambda x: 0])
        hparams.model_type = "RNNDYN-{}x{}_EMB_(-1)-3_RELU_128-2_BiLSTM_32-1_FC_12".format(num_emb, emb_dim)
        model = ModelFactory.create(hparams.model_type, (in_dim,), out_dim, hparams)

        self.assertEqual(1, len(model.emb_groups))
        self.assertEqual(torch.Size([num_emb, emb_dim]), model.emb_groups[0].weight.shape)
        self.assertEqual(torch.Size([128, in_dim - 1 + emb_dim]), model[0].weight.shape)
        self.assertEqual(torch.Size([128, 128 + emb_dim]), model[1].weight.shape)

        self.assertEqual(torch.Size([32 * 4, 128 + emb_dim]), model[3].weight_ih_l0.shape)
        self.assertEqual(torch.Size([32 * 4, 32 * 2 + emb_dim]), model[4].weight_ih_l0_reverse.shape)
        pass

    # def test_compare_to_recursive_matrix(self):
    #     """
    #     Compare the element-wise computed gradient matrix with the recursively generate matrix for alphas in
    #     range(-alpha_range, alpha_range, precision).
    #     """
    #     precision = 0.05  # Precision used for steps in that range.
    #     delta = 0.05  # Allowed delta of error.
    #     alpha_range = 0.2  # Range of alpha to test in.
    #
    #     from idiaptts.src.model_trainers.vtln.VTLNSpeakerAdaptionModelTrainer import VTLNSpeakerAdaptionModelTrainer
    #     from idiaptts.src.neural_networks.pytorch.layers.AllPassWarp import AllPassWarp
    #     hparams = VTLNSpeakerAdaptionModelTrainer.create_hparams()
    #     hparams.out_dir = os.path.join(self.out_dir, "test_compare_to_recursive_matrix")  # Add function name to path.
    #     wl = AllPassWarp(10, hparams)
    #     assert (precision < 2 * alpha_range)  # Precision must fit in range.
    #
    #     for alpha_value in numpy.arange(-alpha_range, alpha_range + precision, precision):
    #         # Alpha value which receives the final gradient.
    #         alpha = torch.tensor(alpha_value, requires_grad=True).type(wl.computation_dtype)
    #         alpha_eps = alpha.repeat([100, 1])  # Test in batch mode.
    #
    #         # Compute the warp matrix for each alpha.
    #         warp_matrix = wl.get_warp_matrix(alpha_eps)
    #
    #         # Create the reference matrix recursively for the given alpha.
    #         ref_matrix = wl.gen_warp_matrix_recursively(alpha)
    #
    #         # Compute the error.
    #         numpy.testing.assert_almost_equal(warp_matrix[10].detach().numpy(), ref_matrix.detach().numpy(), 3)
    #         dist = (warp_matrix[10] - ref_matrix).abs()
    #         max_error = (dist / (ref_matrix.abs() + 1e-6)).max()
    #         # error = dist.sum()
    #         self.assertLess(max_error, delta, msg="Max error between w_matrix_3d and recursive reference is"
    #                                               " {:.5f}% for alpha={:.2f}.".format(max_error * 100, alpha_value))
    #
    #         # Compute the gradient ratio error.
    #         ref_matrix.sum().backward()
    #         real_grad = torch.tensor(alpha.grad)
    #         alpha.grad.zero_()
    #         warp_matrix.sum().backward()
    #         approx_grad = alpha.grad / len(alpha_eps)
    #         dist_grad = (real_grad - approx_grad).abs()
    #         error_ratio = (dist_grad / real_grad.abs())
    #
    #         self.assertLess(error_ratio, delta, msg="Gradient error between w_matrix_3d and recursive reference is "
    #                                                 "{:.5f}% for alpha={:.2f}.".format(error_ratio * 100., alpha_value))
    #
    #     shutil.rmtree(hparams.out_dir, ignore_errors=True)
