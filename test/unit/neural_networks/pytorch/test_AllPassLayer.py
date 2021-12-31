#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#


import unittest

import torch

from idiaptts.src.neural_networks.pytorch.layers.AllPassWarpLayer import AllPassWarpLayer


class TestAllPassLayer(unittest.TestCase):

    def _get_config(self, alpha_layer_in_dims=[4, 2], alpha_ranges=[0.1, 0.2], mean=torch.full((5,), -1, dtype=torch.float32),
                    std_dev=torch.full((5,), 3.0, dtype=torch.float32)):
        return AllPassWarpLayer.Config(alpha_layer_in_dims=alpha_layer_in_dims, alpha_ranges=alpha_ranges,
                                       batch_first=True, warp_matrix_size=5, mean=mean, std_dev=std_dev)

    def test_alpha_layer_generation(self):
        layer = self._get_config().create_model()
        params = list(layer.named_parameters())
        self.assertEqual(4, len(params))

        state = layer.state_dict()
        self.assertEqual(4 + 2, len(state))

    def test_forward(self):
        layer = self._get_config().create_model()
        batch_size = 2
        T = 8
        input_ = torch.rand((batch_size, T, 5))
        alpha_input_1 = torch.rand((batch_size, T, 4))
        alpha_input_2 = torch.rand((batch_size, T, 2))

        output, kwargs = layer((input_, alpha_input_1, alpha_input_2), None, None)
        output, combined_alpha, alpha_1, alpha_2 = output

        self.assertEqual(torch.Size([batch_size, T, 5]), output.shape)
        self.assertEqual(torch.Size([batch_size, T, 1]), combined_alpha.shape)
        self.assertEqual(torch.Size([batch_size, T, 1]), alpha_1.shape)
        self.assertEqual(torch.Size([batch_size, T, 1]), alpha_2.shape)
        self.assertTrue((alpha_1.abs() <= layer.alpha_ranges[0]).all(), msg="Alpha 1 not within allowed range.")
        self.assertTrue((alpha_2.abs() <= layer.alpha_ranges[1]).all(), msg="Alpha 2 not within allowed range.")

    def test_training(self):
        layer = self._get_config().create_model()
        batch_size = 2
        T = 8
        input_ = torch.rand((batch_size, T, 5))
        org_input = input_.detach().clone()
        alpha_input_1 = torch.rand((batch_size, T, 4))
        alpha_input_2 = torch.rand((batch_size, T, 2))

        (output, *_), _ = layer((input_, alpha_input_1, alpha_input_2), None, None)

        layer.alpha_layers[1].weight.retain_grad()
        self.assertFalse(layer.all_pass_warp.w_matrix_3d.requires_grad)

        torch.autograd.set_detect_anomaly(True)
        output.sum().backward()
        self.assertIsNotNone(layer.alpha_layers[1].weight.grad)
        self.assertTrue((input_ == org_input).all())
        self.assertIsNone(layer.mean.grad)
        self.assertIsNone(layer.std_dev.grad)
        self.assertIsNone(layer.all_pass_warp.w_matrix_3d.grad)

    def test_normalisation(self):
        layer = self._get_config().create_model()
        batch_size = 2
        T = 8
        input_ = torch.rand((batch_size, T, 5))
        org_input = input_.clone()
        alpha_input_1 = torch.rand((batch_size, T, 4))
        alpha_input_2 = torch.rand((batch_size, T, 2))

        (output, *_), _ = layer((input_, alpha_input_1, alpha_input_2), None, None)
        self.assertTrue((org_input == input_).all(), msg="Input tensor was changed in-place.")

        layer.mean = None
        layer.std_dev = None
        (no_norm_output, *_), _ = layer((input_, alpha_input_1, alpha_input_2), None, None)

        self.assertFalse(torch.isclose(output, no_norm_output).all(),
                         msg="Normalised warp shouldn't match un-normalised warp.")
