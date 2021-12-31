#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#

# Code from https://github.com/ZackHodari/morgana/

import copy


class ExponentialMovingAverage(object):
    """Exponential moving average helper to apply gradient updates to an EMA model.
    Parameters
    ----------
    model : torch.nn.Module
    decay : float
        Decay rate of previous parameter values. Parameter updates are also scaled by `1 - decay`.
    """
    def __init__(self, model, decay):
        self.model = copy.deepcopy(model)
        self.decay = decay

        # Use shadow to link to all parameters in the averaged model.
        self.shadow = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data
            param.detach_()

    def _update_param(self, name, x):
        """Performs update on one parameter. `shadow = decay * shadow + (1 - decay) * x`."""
        assert name in self.shadow

        update_delta = self.shadow[name] - x
        self.shadow[name] -= (1.0 - self.decay) * update_delta

    def update_params(self, other_model):
        """Updates all parameters of `self.model` using a separate model's updated parameters."""
        assert other_model is not self.model

        for name, param in other_model.named_parameters():
            if name in self.shadow:
                self._update_param(name, param.data)
