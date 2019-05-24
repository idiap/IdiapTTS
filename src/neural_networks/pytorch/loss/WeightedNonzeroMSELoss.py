#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#

import collections
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from torch.nn.modules.loss import _Loss

def weighted_nonzero_mse_loss(input, target, weight_zero, weight_non_zero, reduce=True, size_average=True):
    """
    Weighting the loss of zero and non zero labels differently, the reference are the target labels.

    Math:
    |l for loss
    (1) l_zero = l * zero_occurence
    (2) l_non_zero = l * (1 - zero_occurence)
    (3) l = l_zero + l_non_zero

    (4) l_zero_weighted =!= l_non_zero_weighted
        l_zero * weight_zero =!= l_non_zero * weight_non_zero  |using (1) and (2)
        l * zero_occurence * weight_zero =!= l * (1 - zero_occurence) * weight_non_zero
    => weight_zero = 1 / zero_occurence * k
    => weight_non_zero = 1 / (1 - zero_occurence) * k, with k > 0.

    Insert in (3)
    (5) l =!= l_zero_weighted + l_non_zero_weighted
        l =!= l * zero_occurence * 1 / zero_occurence * k + l * (1 - zero_occurence) * 1 / (1 - zero_occurence) * k
        l =!= l * k + l * k
        l =!= l * 2k
        1 =!= 2k
    => k = 1/2

    :param input:
    :param target:           Is required to be continuous.
    :param weight_zero:
    :param weight_non_zero:
    :param size_average:     If True, sums up the errors of all frames, otherwise returns a vector.
    :return:
    """
    weight_zero = weight_zero / 2
    weight_non_zero = weight_non_zero / 2

    real = ((input - target) ** 2)

    # Create weight tensor.
    if target.is_cuda:
        weight_tensor = torch.cuda.FloatTensor(target.data.nelement()).fill_(weight_zero)
    else:
        weight_tensor = torch.FloatTensor(target.data.nelement()).fill_(weight_zero)
    # Get indices of non-zero elements in target. Note that indices are selected in a 1-D tensor.
    non_zero_indices = target.data.contiguous().view(-1).nonzero().squeeze()
    # print str(non_zero_indices.shape) + " / " + str(target.data.nelement())

    # Add weights for non-zero entries.
    if len(non_zero_indices) != 0:
        weight_tensor.index_fill_(0, non_zero_indices, weight_non_zero)  # Indexing only with 1-D LongTensor.

    # Reshape weight tensor to original dimensions.
    weight_tensor = weight_tensor.view(target.data.shape)
    # print weight_tensor.numpy()[43:48]  # DEBUG

    weight_var = Variable(weight_tensor, requires_grad=False)
    weighted_real = real * weight_var

    real_loss = weighted_real
    if reduce:
        real_loss = real_loss.sum()
        if size_average:
            real_loss = real_loss / input.data.nelement()

    return real_loss


class WeightedNonzeroMSELoss(_Loss):
    r"""
    Creates a criterion that weights the MSE loss of each output differently depending on
    the zero occurrence in the target data. Note that the weights (numpy array) given as
    parameter need to have the same dimension as the NN output per frame.

    Example:
        num_nn_outputs = 9
        non_zero_occurrence = 0.015
        zero_occurence = 1 - non_zero_occurrence

        weights_non_zero = np.zeros((num_nn_outputs))
        weights_non_zero.fill(1 / non_zero_occurrence)
        weights_zero = np.zeros((num_nn_outputs))
        weights_zero.fill(1 / zero_occurence)

        criterion = WeightedNonzeroMSELoss(weights_zero,weights_non_zero, size_average=True)
    """
    def __init__(self, use_gpu, weight_zero, weight_non_zero, reduce=True, size_average=True):
        # TODO: Accept reduction keyword.
        super(WeightedNonzeroMSELoss, self).__init__(size_average=size_average, reduce=reduce)

        self.reduce = reduce
        self.size_average = size_average

        self.register_buffer('weight_zero', torch.tensor(weight_zero))
        self.register_buffer('weight_non_zero', torch.tensor(weight_non_zero))

    def forward(self, input, target):
        return weighted_nonzero_mse_loss(input, target,
                                         self.weight_zero, self.weight_non_zero,
                                         self.reduce,
                                         self.size_average)
