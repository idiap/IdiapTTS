#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#

import os
import sys
import torch
from torch.autograd import Variable
import numpy as np
from numpy import linalg
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from torch.nn.modules.loss import _Loss

if not any(p.endswith("IdiapTTS") for p in sys.path):
    parent_dirs = os.path.realpath(__file__).split(os.sep)
    dir_itts = str.join(os.sep, parent_dirs[:parent_dirs.index("IdiapTTS")+1])
    sys.path.append(dir_itts)  # Adds the IdiapTTS folder to the path, required to work on grid.
from src.DataPlotter import DataPlotter
# from wcad_atom_prediction.data_preparation.AtomLabelGen import AtomLabelGen
from src.neural_networks.pytorch.loss.WeightedNonzeroMSELoss import weighted_nonzero_mse_loss


def atom_loss(input, target, coefs_tensor, integrals_tensor, size_average=True, reduce=True):
    # DEBUG
    # input[:] = 0
    # input[10, 0, 0] = 1
    # input[100, 0, 0] = 1
    # input[100, 0, 0] = 1
    # input[1, 0, 1] = -0.4353
    # input = input[:600]
    # target[:] = 0
    # target[20, 0, 0] = 1
    # target = target[:600]
    # input[:] = 1
    # input[43:45] = -1
    # input[33:43] = -0.005
    # input[44:53] = -0.005
    # input[43] = -0.6
    # input[48] = -1.5
    # input[150:252] = -1

    if len(input.shape) == 2:
        input = input.view(len(input), 1, -1)
        target = target.view(len(target), 1, -1)

    coefs_variable = Variable(coefs_tensor[:len(input), :, :len(input)], requires_grad=False)
    integral_variable = Variable(integrals_tensor[:len(input), :, :len(input)], requires_grad=False)

    input_pos = input[:, 0, :].contiguous()
    input_pos_enveloped_framewise = input_pos.view(len(input), -1, 1) * coefs_variable + 1e-8
    # input_pos_enveloped_framewise = torch.sum(input_pos_enveloped_framewise, dim=0)  # Using sum of columns
    target_pos = target[:, 0, :].contiguous()  # This will do nothing, if target is already properly aligned.
    target_pos_enveloped_framewise = torch.sum(target_pos.view(len(input), -1, 1) * coefs_variable, dim=0)

    error_pos_framewise = (input_pos_enveloped_framewise - target_pos_enveloped_framewise.expand_as(input_pos_enveloped_framewise))
    error_pos_framewise = (error_pos_framewise ** 2) + 1e-8
    error_pos = torch.sum(error_pos_framewise * integral_variable, dim=2)

    # error_amp = input[:, :, :-1] - target[:, :, :-1] # mse_weighted_nonzero_loss(input[:, :, :-1].contiguous().view(len(input), 1, -1), target[:, :, :-1].contiguous().view(len(target), 1, -1), weight_zero, weight_non_zero, size_average=False)
    # #error_amp_plot = (input[:, 0, 0] - target[:, 0, 0])
    # # error_amp_plot = error_amp
    # error_amp = (error_amp ** 2)

    # print("input", input[40:50].squeeze().view(1, -1))
    # print("input_pos_enveloped", input_pos_enveloped[0, 40:50].view(1, -1))
    # print("target", target[40:50].squeeze().view(1, -1))
    # print("target_pos_enveloped", target_pos_enveloped[0, 40:50].view(1, -1))
    # print("weighted_error", weighted_error[42:45, 40:50])
    # # print("integral", integral[40:60].view(1, -1))
    # #print("weighted_real_scaled", weighted_real_scaled[40:60].view(1, -1))
    # print("input_pos_enveloped_framewise", input_pos_enveloped_framewise[42:46, 40:50])
    # print("error_pos_framewise", error_pos_framewise[42:45, 40:50])
    # print("error_pos", error_pos[40:50].view(1, -1))
    # print("error_pos_plot", error_pos_plot[42:45, 40:50])
    # # print("error_mse", error_mse[40:50].squeeze().view(1, -1))
    # # print("weighted_integral", weighted_integral[40:60].view(1, -1))
    # print("real", real[40:50].view(1, -1))

    # if len(input) == 589:
    #     tmp_input = torch.sum(input_pos_enveloped_framewise, dim=0)
    #     tmp_loss = (tmp_input - target_pos_enveloped_framewise)**2*integral_variable
    #     plot_atom_loss(input_pos.data.squeeze().cpu().numpy(),
    #                    target_pos.data.squeeze().cpu().numpy(),
    #                    tmp_input.data.squeeze().cpu().numpy(),
    #                    target_pos_enveloped_framewise.data.cpu().numpy(),
    #                    error_pos.data.cpu().numpy()
    #                    #torch.sum(((tmp_input - target_pos_enveloped_framewise)**2)*integral_variable, dim=2).data.cpu().numpy())

    # l1_loss = ((input - target)**2).sum(dim=2)
    # l1_weight = 0.0
    real_loss = error_pos
    if reduce:
        real_loss = real_loss.sum()
        if size_average:
            real_loss = real_loss / input.data.nelement()

    return real_loss


# def plot_atom_loss(input_pos, target_pos, input_pos_enveloped, target_pos_enveloped, loss):
#
#     if input_pos.ndim == 1:
#         input_pos = input_pos.reshape(-1, 1)
#         target_pos = target_pos.reshape(-1, 1)
#         input_pos_enveloped = input_pos_enveloped.reshape(-1, 1)
#         target_pos_enveloped = target_pos_enveloped.reshape(-1, 1)
#         loss = loss.reshape(-1, 1)
#
#     plotter = DataPlotter()
#     # plotter.set_num_colors(input_pos.shape[1])
#
#     # for idx in range(input_pos.shape[1]):
#     #     plotter.set_data_list(grid_idx=idx,
#     #                           data_list=[(target_pos[:, idx], 'target, theta=' + str(AtomLabelGen.index_to_theta(idx, 0.01, 0.005))),
#     #                                      (input_pos[:, idx], 'NN output')])
#
#     data_input = list()
#     for idx in reversed(range(target_pos.shape[1])):
#         data_input.append((input_pos[:, idx], 'NN out'))
#         #data_input.append((input_pos_enveloped[:, idx], ))
#
#     data_target = list()
#     for idx in reversed(range(input_pos.shape[1])):
#         data_target.append((target_pos[:, idx], 'target'))  # 'target, theta=' + str(AtomLabelGen.index_to_theta(idx, 0.01, 0.005))
#         #data_target.append((target_pos_enveloped[:, idx], ))
#
#     plotter.set_data_list(grid_idx=0, data_list=data_input)
#     plotter.set_data_list(grid_idx=1, data_list=data_target)
#
#     #loss = input_pos - target_pos
#     data_loss = list()
#     data_loss.append((loss, 'loss'))
#     plotter.set_data_list(grid_idx=2, data_list=data_loss)
#
#     plotter.set_label(xlabel='frames', ylabel='amp')
#     #plotter.set_lim(ymin=0, ymax=1.1)
#     #plotter.set_lim(grid_idx=2, ymin=-1.1, ymax=1.1)
#     plotter.gen_plot()
#     plotter.save_to_file("atom_loss_thetas.png")


class AtomLoss(_Loss):
    r"""
    Creates a criterion that learns spike positions. It adds a distribution around each
    spike to compute a temporal-aware MSE loss.
    """
    max_frames = 4000

    def __init__(self, use_gpu, thetas, sum_loss=True, size_average=True):
        super(AtomLoss, self).__init__(size_average=size_average, reduce=sum_loss)

        self.register_buffer('thetas', torch.tensor(thetas))
        self.register_buffer('sum_loss', torch.from_numpy(np.array(1.0)) if sum_loss else None)

        coefs = np.empty((self.max_frames, len(thetas), self.max_frames), dtype=np.float32)
        integrals = np.empty((self.max_frames, len(thetas), self.max_frames), dtype=np.float32)

        from wcad import GammaAtom
        for idx, theta in enumerate(thetas):
            atom = GammaAtom(k=6, theta=theta, fs=1000 / 5, amp=1, position=0)  # TODO: k as parameter, default=6.
            padded_curve = np.roll(atom.get_padded_curve(self.max_frames).astype(np.float32), -1)
            # padded_curve /= padded_curve.max()  # Normalise to size one.
            # padded_curve /= padded_curve.sum()  # Normalise to integral 1 (default).
            padded_curve[padded_curve <= 1e-2] = 0
            integral_len = len(padded_curve[padded_curve > 1e-2])
            # print integral_len  # Print length of the gamme distribution.
            # print padded_curve.sum()  # Print integral of distribution.

            coefs_np_matrix = np.triu(AtomLoss.strided_array(padded_curve), 0)

            # Fill all-zero rows of matrix with a non-zero value.
            # Otherwise the network will not receive any error from those elements corresponding to the all-zero rows.
            # fill_value = padded_curve[np.where(padded_curve != 0)[0][0]]  # Use first non-zero of padded_curve.
            coefs_np_matrix[np.where(~coefs_np_matrix.any(axis=1))[0], -1] = 1
            #coefs_np_matrix /= coefs_np_matrix.sum(axis=1, keepdims=True)  # Each row should sum up to one.

            padded_integral_coefs = np.zeros(self.max_frames).astype(np.float32)
            padded_integral_coefs[:integral_len - 1] = 1.0
            padded_integral_coefs[-1] = 1.0
            integral_np_matrix = np.tril(AtomLoss.strided_array(padded_integral_coefs).transpose(), 0).transpose()

            coefs[:, idx, :] = coefs_np_matrix
            integrals[:, idx, :] = integral_np_matrix

        coefs_tensor = torch.from_numpy(coefs)
        integrals_tensor = torch.from_numpy(integrals)

        if use_gpu:
            coefs_tensor = coefs_tensor.cuda()
            integrals_tensor = integrals_tensor.cuda()

        self.register_buffer('coefs_tensor', coefs_tensor)
        self.register_buffer('integrals_tensor', integrals_tensor)

    def forward(self, input, target):
        _assert_no_grad(target)
        return atom_loss(input, target, self.coefs_tensor, self.integrals_tensor,
                         self.size_average,
                         self.sum_loss)

    @staticmethod
    def strided_array(ar):
        a = np.concatenate((ar, ar[:-1]))
        L = len(ar)
        n = a.strides[0]
        return np.lib.stride_tricks.as_strided(a[L - 1:], (L, L), (-n, n), writeable=False)
