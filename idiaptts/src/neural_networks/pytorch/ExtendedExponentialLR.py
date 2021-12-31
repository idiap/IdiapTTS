#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#

import warnings
import math

from torch.optim.lr_scheduler import ExponentialLR


class ExtendedExponentialLR(ExponentialLR):
    def __init__(self, optimizer, gamma, last_epoch=-1, min_lr=None, warmup_steps=0, decay_steps=1.0):
        if isinstance(min_lr, list) or isinstance(min_lr, tuple):
            if len(min_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} min_lrs, got {}".format(
                    len(optimizer.param_groups), len(min_lr)))
            self.min_lrs = list(min_lr)
        else:
            self.min_lrs = [min_lr] * len(optimizer.param_groups)

        self.warmup_steps = warmup_steps
        self.decay_steps = float(decay_steps)

        super(ExtendedExponentialLR, self).__init__(optimizer, gamma, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch <= self.warmup_steps:
            return self.base_lrs
        return [max(group['lr'] * self.gamma ** (1.0 / self.decay_steps), min_lr)
                for group, min_lr in zip(self.optimizer.param_groups, self.min_lrs)]

    def _get_closed_form_lr(self):
        return [max(base_lr * self.gamma ** max(0, (self.last_epoch - self.warmup_steps + 1) / self.decay_steps), min_lr)
                for base_lr, min_lr in zip(self.base_lrs, self.min_lrs)]
