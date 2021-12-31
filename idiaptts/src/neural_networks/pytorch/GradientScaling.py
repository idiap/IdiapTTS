#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#

'''A gradient scaling layer implemented following https://pytorch.org/docs/master/notes/extending.html'''

import torch


class grad_scaling(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_, lambda_):
        # ctx.set_materialize_grads(False)  # TODO: Add in PyTorch +1.7
        ctx.lambda_ = lambda_
        return input_.view_as(input_)

    @staticmethod
    def backward(ctx, grad_output):
        # Skip computation of empty gradients.
        if grad_output is None:
            return None, None
        # Special case to make sure no gradient is flowing.
        if ctx.lambda_ == 0.0:
            return torch.zeros_like(grad_output), None

        return grad_output * ctx.lambda_, None


class GradientScaling(torch.nn.Module):
    def __init__(self, lambda_: float) -> None:
        super(GradientScaling, self).__init__()
        self.lambda_ = float(lambda_)

    def forward(self, input_):
        return grad_scaling.apply(input_, self.lambda_)

    def extra_repr(self):
        return "lambda={}".format(self.lambda_)
