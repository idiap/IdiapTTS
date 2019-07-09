#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import MSELoss


def weighted_vuv_mse_loss(input, target, value_indices, weighting_decision_index=-1, weight=0.5, decision_index_weight=1.0, size_average=True, reduce=True):
    """
    The column at target[..., weighting_decision_index] is used to weight the MSE loss of all other dimensions.
    The weight is created through (target[..., weighting_decision_index] * (1 - weight)) + weight.
    The MSE loss of the selected column remains unweighted.
        
    :param input:                       Input data.
    :param target:                      Target data.
    :param value_indices:               Indices to select columns in input data which are weighted.
    :param weighting_decision_index:    Index of the column that is used to determine weighting (target[..., weighting_decision_index]).
    :param weight:                      Frames are multiplied by (target[..., weighting_decision_index] * (1 - weight)) + weight.
    :param decision_index_weight:       Weight of the column used for determine the weighting. This weighting is independent from the values in the column.
    :param size_average:                Compute the average of the mean, ignored when reduce=True.
    :param reduce:                      If False, returns the loss per frame with all dimensions.
    :return:                            Average loss, loss, or loss per frame.
    """

    input_areas = input[..., weighting_decision_index:weighting_decision_index + 1]  # Use slicing to preserve 3D.
    # target_values = target[..., value_indices]
    target_areas = target[..., weighting_decision_index:weighting_decision_index + 1]  # Use slicing to preserve 3D.

    # Create the weight tensor.
    weights = (target_areas * (1 - weight)) + weight

    # Compute loss of full data without summing up.
    loss_full = F.mse_loss(input, target, reduction='none')
    # Weight by frame.
    loss_full_weighted = loss_full * weights
    # Compute loss of area output, which is only weighted by the decision_index_weight.
    loss_areas = F.mse_loss(input_areas, target_areas, reduction='none')
    loss_areas = loss_areas * decision_index_weight
    # Apply the loss to the full loss.
    loss_full_weighted[..., weighting_decision_index.long()] = loss_areas

    # Reduce combines all frames.
    if not reduce:
        return loss_full_weighted
    summed_loss = loss_full_weighted.sum()

    # Average loss.
    if size_average:
        return summed_loss / loss_full_weighted.nelement()
    return summed_loss


class WMSELoss(MSELoss):
    """
    The column at target[..., weighting_decision_index] is used to weight the MSE loss of all other dimensions.
    The weight is created through (target[..., weighting_decision_index] * (1 - weight)) + weight.
    The MSE loss of the selection index is only weighted by the decision_index_weight independent from its value.
    """
    def __init__(self, dim_out, weighting_decision_index=-1, weight=0.5, decision_index_weight=1.0, size_average=True, reduce=True):
        """
        Create a new WMSELoss criterion.

        :param dim_out:                     Number of dimension of the output. Used to pre-generate indices.
        :param weighting_decision_index:    The index in the output which should be used as decision for weighting, values in {0, 1}.
        :param weight:                      Weight which is multiplied with the output vector (excluding the weight_index) where the output[weigth_index] is zero.
        :param decision_index_weight:       Multiplied with the loss of the decision index.
        :param size_average:                Average loss over all elements.
        :param reduce:                      Reduce to a single loss value.
        """
        super().__init__(size_average, reduce)
        self.dim_out = dim_out
        self.size_average = size_average
        self.reduce = reduce
        self.register_buffer('weight', torch.Tensor([weight]))
        self.register_buffer('weighting_decision_index', torch.Tensor([weighting_decision_index]).int())
        self.register_buffer('decision_index_weight', torch.Tensor([decision_index_weight]))

        if weighting_decision_index < 0:
            weighting_decision_index += dim_out

        if weighting_decision_index == dim_out - 1:
            value_indices = torch.arange(0, dim_out-1)
        else:
            value_indices = torch.cat((torch.arange(0, weighting_decision_index), torch.arange(weighting_decision_index + 1, dim_out)))

        self.register_buffer('value_indices', value_indices.long())

    def forward(self, input, target):
        assert self.dim_out == target.shape[-1]
        return weighted_vuv_mse_loss(input, target, self.value_indices, self.weighting_decision_index, self.weight, self.decision_index_weight,
                                     self.size_average, self.reduce)

