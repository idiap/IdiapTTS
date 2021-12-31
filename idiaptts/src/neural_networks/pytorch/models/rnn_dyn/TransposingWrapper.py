#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#


from .FFWrapper import FFWrapper


class TransposingWrapper(FFWrapper):
    def __init__(self, in_dim, layer_config, batch_first: bool = True):
        super(TransposingWrapper, self).__init__(in_dim, layer_config)
        self.batch_first = batch_first
        self.transpose = True
        self.untranspose = True

    def forward(self, input_, **kwargs):
        if self.transpose:
            if self.batch_first:
                input_ = input_.transpose(1, 2)
            else:
                input_ = input_.permute(1, 2, 0)

        output, kwargs = super(TransposingWrapper, self).forward(
            input_, **kwargs)

        if self.untranspose:
            if self.batch_first:
                output = output.transpose(1, 2)
            else:
                output = output.permute(2, 0, 1)

        return output, kwargs

    def __getitem__(self, item):
        return self.module.__getitem__(item)
