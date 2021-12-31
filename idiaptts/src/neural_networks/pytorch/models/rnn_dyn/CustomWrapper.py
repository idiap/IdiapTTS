#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#


import torch.nn as nn


class CustomWrapper(nn.Module):

    def __init__(self):
        super(CustomWrapper, self).__init__()
        self.module = None
        self.out_dim = None  # Needs to be set by child classes.

    def forward(self, input_, **kwargs):
        if type(self.module) is nn.Sequential \
                and hasattr(self.module[0], "select_inputs") \
                and callable(self.module[0].select_inputs):
            output = self.module(self.module[0].select_inputs(input_, **kwargs))

        elif hasattr(self.module, "select_inputs") \
                and callable(self.module.select_inputs):
            output = self.module(self.module.select_inputs(input_, **kwargs))

        else:
            output = self.module(input_)

        return output

    def __getattr__(self, item):
        try:
            return super().__getattr__(item)
        except AttributeError as e:
            # TODO: Test speed impact.
            if item != "module":
                return getattr(self.module, item)
            else:
                raise e

    def init_hidden(self, batch_size=1):
        pass
