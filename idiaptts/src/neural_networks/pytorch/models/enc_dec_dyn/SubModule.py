#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#
from idiaptts.src.neural_networks.pytorch.models.NamedForwardModule import NamedForwardModule
import idiaptts.src.neural_networks.pytorch.models.rnn_dyn as rnn_dyn


class SubModule(NamedForwardModule):
    def __init__(self, config):
        super().__init__(input_names=config.input_names,
                         batch_first=True,
                         input_merge_type=config.input_merge_type,
                         name=config.name,
                         output_names=config.output_names)

        if config.config is not None:
            self.model = config.config.create_model()
        else:
            self.model = None

    def forward_module(self, input_, lengths, max_lengths):
        if self.model is None:
            return input_

        if type(self.model) is rnn_dyn.RNNDyn:
            # print("Shape {}, device {}".format(input_.shape, input_.device))
            output, kwargs = self.model(
                input_,
                seq_lengths_input=lengths[self.input_names[0]]
                    if self.input_names is not None else None,
                max_length_inputs=max_lengths[self.input_names[0]]
                    if self.input_names is not None else None)
            # Update output lengths. Assumes that all outputs have the
            # same length.
            lengths.update({name: kwargs["seq_lengths_input"]
                            for name in self.output_names})
            max_lengths.update({name: kwargs["max_length_inputs"]
                                for name in self.output_names})
        else:
            # This call expects models to change the lengths and
            # max_lengths dictionaries internally.
            output = self.model(input_, lengths, max_lengths)
            for name in self.output_names:
                assert name in lengths, "Sequence length for output {} of {} "\
                    " was not added.".format(name, self.model)
                assert name in max_lengths, "Max length for output {} of {} " \
                    "was not added".format(name, self.model)

        return output

    def init_hidden(self, batch_size=1):
        if self.model is not None:
            self.model.init_hidden(batch_size)
