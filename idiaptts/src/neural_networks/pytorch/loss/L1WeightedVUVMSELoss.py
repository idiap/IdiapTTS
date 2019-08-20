#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#

import torch

from idiaptts.src.neural_networks.pytorch.loss.WMSELoss import WMSELoss, weighted_mse_loss


class L1WeightedVUVMSELoss(WMSELoss):
    def __init__(self, weight_unvoiced=0.5, vuv_loss_weight=1, size_average=True, reduce=True, L1_loss_weight=1, vector_loss=False):
        """

        :param weight_unvoiced:       Weight of loss at unvoiced frames, voiced frames weight is 1.
        :param vuv_loss_weight:       Weighting of loss on V/UV flag (decision_index_weight).
        :param size_average:          Return mean over time, only used when reduce=True.
        :param reduce:                Sum over time.
        :param L1_loss_weight:        Weighting of L1 loss on spiking inputs.
        :param vector_loss:           Return the three losses stacked in the first dimension. Reduce and size_average apply as before.
        """
        super().__init__(dim_out=2,
                         weighting_decision_index=1,
                         weight=weight_unvoiced,
                         decision_index_weight=vuv_loss_weight,
                         size_average=size_average,
                         reduce=reduce)

        self.register_buffer('L1_loss_weight', torch.Tensor([L1_loss_weight]).squeeze_())
        self.vector_loss = vector_loss

    def forward(self, input, target):
        lf0_vuv_input = input[..., 0:2]
        amps_input = input[..., 2:]

        lf0_vuv_loss = super().forward(lf0_vuv_input, target)
        lf0_loss = lf0_vuv_loss[..., 0]
        vuv_loss = lf0_vuv_loss[..., 1]
        L1_constraint = amps_input.abs().mean(dim=-1)

        # TODO THIS IS A DIRTY HACK it works but... what with non zero weight? (see the hacky part with the mask)
        lf0_loss.div_(target[..., 1].sum(dim=0))
        ##########################################################################

        if self.vector_loss:
            if not self.reduce:  # Ignore size_average.
                return torch.cat((lf0_loss[None, ...], vuv_loss[None, ...], L1_constraint[None, ...]))
            if not self.size_average:
                return torch.cat((lf0_loss.sum(dim=0), vuv_loss.sum(dim=0), L1_constraint.sum(dim=0)))
            else:
                return torch.cat((lf0_loss.mean(dim=0), vuv_loss.mean(dim=0), L1_constraint.mean(dim=0)))
        else:
            L1_constraint.mul_(self.L1_loss_weight)

            if not self.reduce:
                return torch.cat((lf0_vuv_loss, L1_constraint.unsqueeze(-1)), dim=-1)
            if not self.size_average:
                return lf0_loss.sum() + vuv_loss.sum() + L1_constraint.sum()
            else:
                return lf0_loss.mean() + vuv_loss.mean() + L1_constraint.mean()

