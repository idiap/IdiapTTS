#
# Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#


import unittest

import os
import shutil
import jsonpickle

import numpy
import torch
from torch import nn

from idiaptts.misc.utils import makedirs_safe
from idiaptts.src.model_trainers.ModularTrainer import ModularTrainer
import idiaptts.src.neural_networks.pytorch.models.rnn_dyn as rnn_dyn


class TestRNNDyn(unittest.TestCase):

    out_dir = None

    @classmethod
    def setUpClass(cls):
        cls.out_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), type(cls()).__name__)
        makedirs_safe(cls.out_dir)  # Create class name directory.

    @classmethod
    def tearDownClass(cls):
        os.rmdir(cls.out_dir)  # Remove class name directory, should be empty.

    def _f_get_emb_index(self):
        return 0

    def test_nonlins(self):
        hparams = ModularTrainer.create_hparams()
        in_dim = 42
        out_dim = 12
        # hparams.model_type = "RNNDYN-1_FC_16-1_LIN_18-1_linear_20-1_RELU_22-1_TANH_24-1_FC_{}".format(out_dim)
        model_config = rnn_dyn.Config(in_dim=in_dim, batch_first=True, layer_configs=[
            rnn_dyn.Config.LayerConfig(layer_type="FC", out_dim=16),
            rnn_dyn.Config.LayerConfig(layer_type="LIN", out_dim=18),
            rnn_dyn.Config.LayerConfig(layer_type="linear", out_dim=20),
            rnn_dyn.Config.LayerConfig(layer_type="Linear", num_layers=2, out_dim=22, nonlin="ReLU"),
            rnn_dyn.Config.LayerConfig(layer_type="Linear", out_dim=22),
            rnn_dyn.Config.LayerConfig(layer_type="SELU", inplace=True),
            rnn_dyn.Config.LayerConfig(layer_type="Linear", out_dim=out_dim),
            rnn_dyn.Config.LayerConfig(layer_type="Conv1d", kernel_size=5, nonlin="ReLU", out_dim=out_dim)
        ], hparams=hparams)
        model = model_config.create_model()
        # print(list(model.modules()))
        # model = ModelFactory.create(hparams.model_type, (in_dim,), out_dim, hparams)

        for layer_idx in range(3):
            num_sublayers = len(model[layer_idx].module)
            if num_sublayers > 1:
                self.assertEqual(1, num_sublayers,
                                 "Layer {} should not have a non linearity but has {}.".format(layer_idx, type(model[layer_idx].module[1])))
        seq_layer = model[3].module
        self.assertEqual(torch.nn.ReLU, type(seq_layer[1]),
                         "Layer {} should have a non-linearity {} but has {}.".format(3, torch.nn.ReLU, type(seq_layer[1])))
        self.assertEqual(torch.nn.ReLU, type(seq_layer[3]),
                         "Layer {} should have a non-linearity {} but has {}.".format(3, torch.nn.ReLU, type(seq_layer[1])))
        layer = model[5].module[0]
        self.assertEqual(torch.nn.SELU, type(layer),
                         "Layer {} should be {} but is {}.".format(5, torch.nn.SELU, type(layer)))
        seq_layer = model[7].module
        self.assertEqual(torch.nn.ReLU, type(seq_layer[1]),
                         "Layer {} should have a non-linearity {} but has {}.".format(3, torch.nn.ReLU, type(seq_layer[1])))

    def test_legacy_string_conversion(self):
        hparams = ModularTrainer.create_hparams()
        num_emb = 3
        emb_dim = 12
        in_dim = 43  # Includes embedding index.
        out_dim = 12
        hparams.add_hparam("f_get_emb_index", [self._f_get_emb_index])
        hparams.model_type = "RNNDYN-{}x{}_EMB_(0, 3, 5)-2_RELU_128-1_Batch" \
            "Norm1dConv1d_18_3-1_BiLSTM_32-1_RNNTANH_8-1_FC_{}".format(
                num_emb, emb_dim, out_dim)
        model = rnn_dyn.convert_legacy_to_config(
            in_dim=(in_dim,), hparams=hparams).create_model()

        self.assertEqual(torch.Size([128, 42 + emb_dim]),
                         model[0][0].weight.shape)
        self.assertEqual(torch.Size([128, 128]),
                         model[0][2].weight.shape)
        self.assertEqual(nn.BatchNorm1d, type(model[2][0]))
        self.assertEqual(torch.Size([4 * 32, 18 + emb_dim]),
                         model[3].module.weight_ih_l0.shape)
        self.assertEqual('RNN_TANH', model[4].module.mode)
        self.assertEqual(torch.Size([12, 8 + emb_dim]),
                         model[5][0].weight.shape)

        seq_length = torch.tensor((100, 75), dtype=torch.long)
        batch_size = 2
        test_input = torch.ones([seq_length[0], batch_size, in_dim])
        model.init_hidden(batch_size)
        output = model(test_input, seq_lengths_input=seq_length,
                       max_length_inputs=seq_length[0])
        self.assertEqual(torch.Size([seq_length[0], batch_size, out_dim]),
                         output[0].shape)

    def test_embeddings(self):
        hparams = ModularTrainer.create_hparams()
        num_emb = 3
        emb_dim = 12
        in_dim = 42  # Contains the embedding index.
        out_dim = 12
        model_config = rnn_dyn.Config(
            in_dim=in_dim,
            layer_configs=[
                rnn_dyn.Config.LayerConfig(layer_type="FC", out_dim=128,
                                           num_layers=2, nonlin="relu"),
                rnn_dyn.Config.LayerConfig(layer_type="FC", out_dim=128,
                                           num_layers=3, nonlin="tanh"),
                rnn_dyn.Config.LayerConfig(layer_type="LSTM", out_dim=32,
                                           num_layers=3, bidirectional=True),
                rnn_dyn.Config.LayerConfig(layer_type="FC", out_dim=out_dim)
            ],
            emb_configs=[
                rnn_dyn.Config.EmbeddingConfig(
                    embedding_dim=emb_dim,
                    name="emb1",
                    num_embedding=num_emb,
                    affected_layer_group_indices=(0, 2, 3))
            ])
        model = model_config.create_model()
        hparams.add_hparam("f_get_emb_index", [self._f_get_emb_index])

        self.assertEqual(1, len(model.emb_groups))
        self.assertEqual(torch.Size([num_emb, emb_dim]),
                         model.emb_groups["emb1"].weight.shape)
        self.assertEqual(torch.Size([128, in_dim + emb_dim]),
                         model[0][0].weight.shape)
        self.assertEqual(torch.Size([128, 128]), model[0][2].weight.shape)
        self.assertEqual(torch.Size([128, 128]), model[1][0].weight.shape)
        self.assertEqual(torch.nn.Tanh, type(model[1][1]))

        self.assertEqual(torch.Size([32 * 4, 128 + emb_dim]),
                         model[2].weight_ih_l0.shape)
        self.assertEqual(torch.Size([32 * 4, 32 * 2]),
                         model[2].weight_ih_l2_reverse.shape)

        seq_length = torch.tensor((100, 75), dtype=torch.long)
        batch_size = 2
        test_input = torch.ones([batch_size, seq_length[0], in_dim])
        test_input_emb = torch.ones([batch_size, seq_length[0], 1])
        model.init_hidden(batch_size)
        output = model(test_input, test_input_emb,
                       seq_lengths_input=seq_length,
                       max_length_inputs=seq_length[0])
        self.assertEqual(torch.Size([batch_size, seq_length[0], out_dim]),
                         output[0].shape)

        seq_length = torch.tensor((100,), dtype=torch.long)
        batch_size = 1
        test_input = torch.ones([batch_size, seq_length[0], in_dim])
        test_input_emb = torch.ones([batch_size, seq_length[0], 1])
        model.init_hidden(batch_size)
        output = model(test_input, test_input_emb,
                       seq_lengths_input=seq_length,
                       max_length_inputs=seq_length[0])
        self.assertEqual(torch.Size([batch_size, seq_length[0], out_dim]),
                         output[0].shape)

    def test_embeddings_everywhere(self):
        hparams = ModularTrainer.create_hparams()
        num_emb = 3
        emb_dim = 12
        in_dim = 43
        out_dim = 12
        hparams.add_hparam("f_get_emb_index", [self._f_get_emb_index])
        hparams.model_type = "RNNDYN-{}x{}_EMB_(-1)-3_RELU_128-2_BiLSTM_32-1_FC_12".format(num_emb, emb_dim)
        model = rnn_dyn.convert_legacy_to_config(in_dim=(in_dim,), hparams=hparams).create_model()

        self.assertEqual(1, len(model.emb_groups))
        self.assertEqual(torch.Size([num_emb, emb_dim]), model.emb_groups["0"].weight.shape)
        self.assertEqual(torch.Size([128, in_dim - 1 + emb_dim]), model[0][0].weight.shape)
        self.assertEqual(torch.Size([12, 64 + emb_dim]), model[2][0].weight.shape)

        self.assertEqual(torch.Size([32 * 4, 128 + emb_dim]), model[1].weight_ih_l0.shape)
        self.assertEqual(torch.Size([32 * 4, 32 * 2]), model[1].weight_ih_l1_reverse.shape)
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

    def test_conv1d(self):
        hparams = ModularTrainer.create_hparams()
        in_dim = 40
        out_dim = 12
        hparams.model_type = "RNNDYN-" + "-".join(["1_BatchNorm1dConv1d_128_5"] * 2) + "-1_BiLSTM_8-1_FC_12"
        model = rnn_dyn.convert_legacy_to_config(in_dim=in_dim, hparams=hparams).create_model()
        #  ModelFactory.create(hparams.model_type, (in_dim,), out_dim, hparams)

        self.assertEqual(in_dim, model[0][0].in_channels)
        self.assertEqual(128, model[0][0].out_channels)
        self.assertEqual((5,), model[0][0].kernel_size)
        for idx in range(1, 4, 2):  # Test for batch norm after each layer.
            self.assertEqual(torch.nn.BatchNorm1d, type(model[idx][0]))

        seq_length = torch.tensor((100, 75), dtype=torch.long)
        batch_size = 2
        test_input = torch.ones([seq_length[0], batch_size, in_dim])
        model.init_hidden(batch_size)
        output = model(test_input, seq_lengths_input=seq_length, max_length_inputs=seq_length[0])
        self.assertEqual(torch.Size([seq_length[0], batch_size, out_dim]), output[0].shape)

        hparams.model_type = "RNNDYN-2_Conv1d_128_5x1-1_FC_12"
        model = rnn_dyn.convert_legacy_to_config(in_dim=in_dim, hparams=hparams).create_model()
        self.assertEqual((5, 1), model[0][0].kernel_size)

        hparams.model_type = "RNNDYN-2_Conv1d_128_5x1_s2_p5_d3_g4-1_FC_12"
        model = rnn_dyn.convert_legacy_to_config(in_dim=in_dim, hparams=hparams).create_model()
        self.assertEqual((2,), model[0][0].stride)
        self.assertEqual((5,), model[0][0].padding)
        self.assertEqual((3,), model[0][0].dilation)
        self.assertEqual(4, model[0][0].groups)

        hparams.model_type = "RNNDYN-2_Conv1d_64_3_p0_s2"
        model = rnn_dyn.convert_legacy_to_config(in_dim=in_dim, hparams=hparams).create_model()
        model.init_hidden(batch_size)
        output, kwargs = model(test_input, seq_lengths_input=seq_length, max_length_inputs=seq_length[0])

        def new_lengths(x):
            return (x - 3) // 2 + 1

        expected_seq_lengths = new_lengths(new_lengths(seq_length))
        expected_max_length = new_lengths(new_lengths(seq_length.max()))
        self.assertTrue((expected_seq_lengths == kwargs["seq_lengths_input"]).all())
        self.assertTrue((expected_max_length == kwargs["max_length_inputs"]).all())

    def test_save_load(self):
        num_emb = 3
        emb_dim = 12
        in_dim = 42  # Contains the embedding index.
        out_dim = 12
        model_config = rnn_dyn.Config(in_dim=in_dim, layer_configs=[
            rnn_dyn.Config.LayerConfig(layer_type="FC", out_dim=128, num_layers=2, nonlin="relu"),
            rnn_dyn.Config.LayerConfig(layer_type="FC", out_dim=128, num_layers=3, nonlin="tanh"),
            rnn_dyn.Config.LayerConfig(layer_type="LSTM", out_dim=32, num_layers=3, bidirectional=True),
            rnn_dyn.Config.LayerConfig(layer_type="FC", out_dim=out_dim)
        ],
        emb_configs=[
            rnn_dyn.Config.EmbeddingConfig(embedding_dim=emb_dim, name="emb1", num_embedding=num_emb,
                                          affected_layer_group_indices=(0, 2, 3))
        ])
        model = model_config.create_model()

        other_model = model_config.create_model()
        self.assertTrue((list(model.parameters())[0] != list(other_model.parameters())[0]).any())

        config_json = model.get_config_as_json()
        params = model.state_dict()
        recreated_config = jsonpickle.decode(config_json)
        recreated_model = recreated_config.create_model()
        recreated_model.load_state_dict(params)

        self.assertTrue((list(model.parameters())[0] == list(recreated_model.parameters())[0]).all())
