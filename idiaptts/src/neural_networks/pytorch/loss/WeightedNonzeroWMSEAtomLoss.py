#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#

# System imports.
import os
import sys
import numpy as np
from numpy import linalg
from torch.legacy.nn import Criterion
import torch.nn.functional as F
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

# Third-party imports.
import torch
from torch.autograd import Variable
from torch.nn.modules.loss import _Loss

# Local source tree imports.
from idiaptts.src.neural_networks.pytorch.loss.WeightedNonzeroMSELoss import weighted_nonzero_mse_loss
from idiaptts.src.neural_networks.pytorch.loss.AtomLoss import AtomLoss, atom_loss
# from wcad_atom_prediction.DataPlotter import DataPlotter


# def plot_mse_with_atom_loss(input_pos_flag, target_pos_flag, input_amps, target_amps):
#
#     plotter = DataPlotter()
#     plotter.set_num_colors(input_amps.shape[1])
#
#     data = list()
#     data.append((input_pos_flag, 'NN output'))
#     data.append((target_pos_flag, 'target'))
#     plotter.set_data_list(grid_idx=0, data_list=data)
#     # plotter.set_lim(ymin=-1, ymax=1)
#
#     data_amps = list()
#     for idx in range(input_amps.shape[1]):
#         data_amps.append((input_amps[:, idx],))
#     for idx in range(target_amps.shape[1]):
#         data_amps.append((target_amps[:, idx], "theta=" + str(AtomLabelGen.index_to_theta(idx, 0.01, 0.005))))
#     plotter.set_data_list(grid_idx=1, data_list=data_amps)
#
#     plotter.set_label(xlabel='frames', ylabel='amp')
#     plotter.gen_plot()
#     plotter.save_to_file("mse_with_atom_loss.png")


class WeightedNonzeroWMSEAtomLoss(AtomLoss):
    r"""
    This criterion is a subclass of AtomLoss that computes the loss of the last element of each frame by
    atom_loss and the rest by a weighted MSE loss. The non-zero weight is applied elementwise.
    """

    def __init__(self, use_gpu, theta, weights_vuv, weights_zero, weights_non_zero, size_average=True, reduce=True):
        super(WeightedNonzeroWMSEAtomLoss, self).__init__(use_gpu, [theta], sum_loss=reduce, size_average=size_average)

        # TODO: Accept reduction keyword.
        self.reduce = reduce
        self.size_average = size_average
        self.register_buffer('weights_vuv', torch.tensor(weights_vuv, dtype=torch.float32))
        self.register_buffer('weights_zero', torch.tensor(weights_zero, dtype=torch.float32))
        self.register_buffer('weights_non_zero', torch.tensor(weights_non_zero, dtype=torch.float32))

    def forward(self, input, target):

        # tmp_all = input[100, :, :]
        # tmp_amps = input[100, :, :-1]
        # tmp_pos = input[100, :, -1:]
        error_pos_flag = atom_loss(input[:, :, -1:], target[:, :, -1:], self.coefs_tensor, self.integrals_tensor,
                                   reduce=False)

        error_amps = weighted_nonzero_mse_loss(input[:, :, 1:-1],  # Select only amplitudes; last element is pos flag.
                                               target[:, :, 1:-1].contiguous(),
                                               self.weights_zero,
                                               self.weights_non_zero,
                                               reduce=False)

        # error_combined = torch.cat((error_amps, error_pos_flag.unsqueeze_(1)), 2)

        target_vuv = target[:, :, 0]  # VUV=1 means voiced.
        output_vuv = input[:, :, 0]
        error_vuv = (output_vuv - target_vuv)**2

        vuv_weight = self.weights_vuv
        vuv_weight_tensor = vuv_weight + (1 - vuv_weight) * target_vuv
        error_pos_flag = error_pos_flag * vuv_weight_tensor
        error_amps = torch.unsqueeze(vuv_weight_tensor, -1) * error_amps

        # if self.reduce is not None:
        #     error_combined = error_vuv.sum() + error_amps.sum() + error_pos_flag.sum()
        #     if self.size_average:
        #         error_combined = error_combined / input.data.nelement()
        if self.reduce:
            if self.size_average:
                error_combined = (error_vuv.mean() + error_amps.mean() + error_pos_flag.mean())# / 3.0
            else:
                error_combined = error_vuv.sum() + error_amps.sum() + error_pos_flag.sum()
        else:
            error_combined = torch.cat((error_vuv.unsqueeze_(-1), error_amps, error_pos_flag.unsqueeze_(-1)), 2)

        # if len(input) == 589:
        #     plot_mse_with_atom_loss(input[:, :, -1].data.squeeze().cpu().numpy(),
        #                             target[:, :, -1].data.cpu().numpy(),
        #                             input.data.cpu().numpy()[:, 0, :-1],
        #                             target.data.cpu().numpy()[:, 0, :-1])

        return error_combined
