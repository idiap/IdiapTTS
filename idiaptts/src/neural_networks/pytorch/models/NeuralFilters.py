#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#

# System imports.
import sys
import numpy as np
import os
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

# Third-party imports.

# Local source tree imports.
from idiaptts.src.neural_networks.pytorch.ModelHandlerPyTorch import ModelHandlerPyTorch
from idiaptts.src.neural_networks.pytorch.models.IntonationFilters import ComplexModel, CriticalModel, modulus_to_theta


class NeuralFilters(nn.Module):
    IDENTIFIER = "NeuralFilters"

    def __init__(self, dim_in, dim_out, hparams):
        super().__init__()

        # Store parameters.
        self.use_gpu = hparams.use_gpu
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dropout = hparams.dropout

        assert(not hparams.batch_first)  # This implementation doesn't work with batch_first.

        self.model_handler_atoms = ModelHandlerPyTorch()
        self.model_handler_atoms.load_checkpoint(hparams.atom_model_path, hparams.hparams_atom, hparams.hparams_atom.learning_rate)
        self.add_module("atom_model", self.model_handler_atoms.model)  # Add atom model as submodule so that parameters are properly registered.

        if hparams.complex_poles:
            self.intonation_filters = ComplexModel(hparams.thetas, hparams.phase_init)
        else:
            self.intonation_filters = CriticalModel(hparams.thetas)
        self.add_module("intonation_filters", self.intonation_filters)

    def forward(self, inputs, hidden, seq_lengths, max_lenght_inputs, *_):
        output_atoms, output_atoms_hidden = self.model_handler_atoms.model(inputs, hidden, seq_lengths, max_lenght_inputs)

        vuv = output_atoms[:, :, 0:1]
        amps = output_atoms[:, :, 1:-1]
        # pos = output_atoms[:, :, -1]

        if len(seq_lengths) > 1:
            pack_amps = pack_padded_sequence(amps, seq_lengths)  # TODO: Add batch_first parameter.

            output_filters = self.intonation_filters(pack_amps)  # The filter unpacks the sequence.

            # output, _ = pad_packed_sequence(output_filters, total_length=max_lenght_inputs)
            # # Pack sequence.
            # pack_amps = amps.squeeze().split(seq_lengths, dim=0)
            # pack_amps = pack_sequence(pack_amps)
            # # Run through filter.
            # output_filters = self.intonation_filters(pack_amps)
            # # Unpack sequence.
            # # output_filters, _ = pad_packed_sequence(output_filters)
            # output_filters = torch.cat([x[:seq_lengths[i], :] for i, x in enumerate(output_filters.split(1, dim=1))])
        else:
            output_filters = self.intonation_filters(amps)

        output_e2e = torch.cat((output_filters, vuv, amps), -1)

        return output_e2e, None

    def filters_forward(self, inputs, hidden, seq_lengths, max_length):
        """Get output of each filter without their superposition."""
        output_atoms, output_atoms_hidden = self.model_handler_atoms.model(inputs, hidden, seq_lengths, max_length)

        amps = output_atoms[:, :, 1:-1]

        if len(seq_lengths) > 1:
            # Pack sequence.
            pack_amps = pack_padded_sequence(amps, seq_lengths)  # TODO: Add batch_first parameter.
            # Run through filter.
            output_filters = self.intonation_filters.filters_forward(pack_amps)  # The filter unpacks the sequence.
        else:
            output_filters = self.intonation_filters.filters_forward(amps)

        return output_filters

    def set_gpu_flag(self, use_gpu):
        self.use_gpu = use_gpu
        self.model_handler_atoms.use_gpu = use_gpu
        self.model_handler_atoms.model.set_gpu_flag(use_gpu)

    def init_hidden(self, batch_size=1):
        self.model_handler_atoms.model.init_hidden(batch_size)
        return None

    def thetas_approx(self):
        roots = [np.roots(denom) for denom in self.intonation_filters.filters.denominator]
        modulus = np.array([np.abs(root[0]) for root in roots])
        return modulus_to_theta(modulus)

