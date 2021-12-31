#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#

# Test cases inspired by PyTorch test cases https://github.com/pytorch/pytorch/blob/5b0f40048899e398d286fe7b55f297991f93ba2c/test/test_optim.py


import unittest

import torch
from torch import nn

from idiaptts.src.neural_networks.pytorch.GradientScaling import GradientScaling


class TestGradientScaling(unittest.TestCase):
    in_dim = 4
    lin_layer = nn.Linear(in_dim, 3)

    def _get_network(self, lambda_=None):
        layers = [nn.Linear(self.in_dim, 3)]
        layers[0].weight = self.lin_layer.weight
        layers[0].bias = self.lin_layer.bias
        if lambda_ is not None:
            layers.append(GradientScaling(lambda_))
        return nn.Sequential(*layers)

    def test_scaling(self):
        scaling = 5
        batch_size = 2

        net = self._get_network()
        net_with_scaling = self._get_network(lambda_=scaling)
        input_ = torch.randn((batch_size, self.in_dim), requires_grad=True)
        input_scaled = input_.detach().clone()
        input_scaled.requires_grad = True

        output = net(input_)
        output.sum().backward()
        output_scaled = net_with_scaling(input_scaled)
        net_with_scaling[0].weight = net[0].weight
        output_scaled.sum().backward()

        self.assertTrue(torch.isclose(scaling * input_.grad, input_scaled.grad).all())

    def test_stop(self):
        batch_size = 2

        net_with_grad_stop = self._get_network(lambda_=0.0)
        input_ = torch.randn((batch_size, self.in_dim), requires_grad=True)

        output_scaled = net_with_grad_stop(input_)
        output_scaled.sum().backward()

        self.assertTrue((input_.grad == 0).all())
