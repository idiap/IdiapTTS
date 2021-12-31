#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#

import torch
from torch.nn.modules.loss import _Loss

from idiaptts.src.neural_networks.pytorch.loss.NamedLoss import NamedLoss


class VAEKLDLoss(NamedLoss):

    class Config(NamedLoss.Config):
        def __init__(self,
                     name,
                     input_names,
                     annealing_steps=200,
                     annealing_points=(25000, 150000),  # TODO: Use 400, (20000, 60000) for CopyCat
                     batch_first=True,
                     input_merge_type=NamedLoss.Config.MERGE_TYPE_LIST,
                     seq_mask=None,
                     start_step=0,
                     reduction='mean_per_frame',
                     **kwargs):

            if "type_" in kwargs:
                kwargs.pop("type_")

            super().__init__(name, type_="VAEKLDLoss", seq_mask=seq_mask, input_names=input_names,
                             batch_first=batch_first, input_merge_type=input_merge_type, start_step=start_step,
                             reduction=reduction, **kwargs)

            assert annealing_steps >= 0, "Annealing steps must be greater or equal zero."
            self.annealing_steps = annealing_steps
            assert annealing_points[0] <= annealing_points[1], \
                "First annealing point must be smaller than second annealing point."
            self.annealing_points = annealing_points

        def create_loss(self):
            return VAEKLDLoss(self)

    def __init__(self, config):
        super().__init__(config)

        self._annealing_steps = config.annealing_steps
        self._annealing_points = config.annealing_points  # TODO: Different in distributed training, divide by number of machines.

    def forward(self, data, length_dict, step):
        loss_dict = super().forward(data, length_dict, step)

        return {k: self._anneal(v, step) for k, v in loss_dict.items()}

    def loss_fn(self, mu, log_var):
        kl_loss = 0.5 * (torch.exp(log_var) + mu**2 - 1. - log_var).sum(dim=-1, keepdim=True)
        return kl_loss

    def _anneal(self, loss, step):
        if step % self._annealing_steps == 0 and step > self._annealing_points[0]:
            if step > self._annealing_points[1]:
                annealing_factor = 1.0
            else:
                annealing_factor = (step - self._annealing_points[0]) / (self._annealing_points[1] - self._annealing_points[0])
        else:
            annealing_factor = 0.0
        return loss * annealing_factor  # * self._weight TODO: Add weights to all losses -> requires a loss base class
