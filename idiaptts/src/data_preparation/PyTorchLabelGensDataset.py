#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#

"""Module description:
   Implements a PyTorch Dataset class that works with two LabelGen classes.
"""

# System imports.
import numpy as np

# Third-party imports.
import torch
from torch.utils.data import Dataset


class PyTorchLabelGensDataset(Dataset):
    """Dataset that generate the samples from two LabelGen objects."""

    def __init__(self, id_list, label_gen_in, label_gen_out, hparams, label_gens_extra=None,
                 match_lengths=False, len_in_out_multiplier=1, random_select=False, max_frames_input=-1):
        """
        Initialise a dataset that generate the samples from two LabelGen objects.

        :param id_list:                 List of ids used in this dataset.
        :param label_gen_in:            LabelGen for input labels.
        :param label_gen_out:           LabelGen for output labels.
        :param hparams:                 Hyper-parameter container.
        :param label_gens_extra:        A list of label generators that create additional components of the batch.
                                        Used e.g. to add a ground truth attention matrix.
        :param match_lengths:           Check lengths of input and output labels match. Guaranteed for random_select.
        :param len_in_out_multiplier:   Multiplier of length between input and output labels in time (can be < 1).
        :param random_select:           Randomly selects a sequential part of the labels. This avoids memory issues.
                                        If True, max_frames_input has to be > 0 and length_check is implicit.
        :param max_frames_input:        Number of frames selected from input labels, has to be a positive integer.
        """
        self.id_list = id_list

        self.LabelGenIn = label_gen_in
        self.LabelGenOut = label_gen_out
        self.LabelGensExtra = label_gens_extra if type(label_gens_extra) in [list, tuple] else (label_gens_extra,) if label_gens_extra is not None else tuple()

        # Automatically add embedding indices to the inputs with given function(s).
        if hparams is not None and hasattr(hparams, "f_get_emb_index"):
            self.f_get_emb_index = hparams.f_get_emb_index
        else:
            self.f_get_emb_index = None

        # Select the appropriate getitem method.
        if random_select:
            assert(max_frames_input >= 1)  # When random_select is used a maximum number of input frames is required.
            self.fun_getitem = self.getitem_random_select
        elif match_lengths:
            self.fun_getitem = self.getitem_match_lengths
        else:
            self.fun_getitem = self.getitem_no_length_check

        self.len_in_out_multiplier = len_in_out_multiplier
        assert(type(max_frames_input) is int)  # Maximum number of frames must be an integer.
        self.max_frames_input = max_frames_input

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, item):
        return self.fun_getitem(self.id_list[item], load_target=True)

    def getitem_by_name(self, id_name, load_target):
        return self.getitem_no_length_check(id_name, load_target)

    def getitem_match_lengths(self, id_name, load_target=True):
        """
        Load labels and trim to match lengths. The self.len_in_out_multiplier is considered here.
        Ensures: len(labels_in) * self.len_in_out_multiplier = len(labels_out)
        self.len_in_out_multiplier can be any positive value.
        """

        labels_in, labels_out, *labels_extra = self.getitem_no_length_check(id_name, load_target)

        # Trim the in labels so that they are dividable by the scale factor.
        if self.len_in_out_multiplier < 1:
            remainder = int(len(labels_in) % (1.0 / self.len_in_out_multiplier))
            if remainder > 0:
                labels_in = labels_in[:-remainder]

        if load_target:
            # Trim the out labels so that they are dividable by the scale factor.
            if self.len_in_out_multiplier > 1:
                remainder = int(len(labels_out) % self.len_in_out_multiplier)
                if remainder > 0:
                    labels_out = labels_out[:-remainder]

            len_diff = len(labels_in) - len(labels_out) // self.len_in_out_multiplier
            if len_diff > 0:  # Input longer than output; trim input.
                trim_front = len_diff // 2
                trim_end = len_diff - trim_front
                labels_in = self.LabelGenIn.trim_end_sample(labels_in, trim_end)
                labels_in = self.LabelGenIn.trim_end_sample(labels_in, trim_front, reverse=True)
            elif len_diff < 0:  # Output longer than input; trim output.
                trim_front = abs(len_diff) // 2
                trim_end = abs(len_diff) - trim_front
                labels_out = self.LabelGenOut.trim_end_sample(labels_out, trim_end)
                labels_out = self.LabelGenOut.trim_end_sample(labels_out, trim_front, reverse=True)

            assert(len(labels_in) == len(labels_out))

        # NOTE: labels_extra shouldn't need any length check. If they do they should be part of labels_in or labels_out.
        return (labels_in, labels_out, *labels_extra)

    def getitem_random_select(self, id_name, load_target=True):
        """
        Randomly selects a part of the input and corresponding output labels.
        The selection is done in the input space. self.max_frames_input frames are used.
        The start frame in the output labels is determined by selected_input_frame * self.len_in_out_multiplier.
        It is ensured that num_out_frames is dividable by self.len_in_out_multiplier.
        """

        labels_in, labels_out, *labels_extra = self.getitem_no_length_check(id_name, load_target)

        # Randomly select a subset of the data. Use torch as random number generator to make results reproducible.
        start_frame_in = torch.IntTensor(1).random_(0, max(1, len(labels_in) - self.max_frames_input))
        # Check if data is shorter than max_frames_input.
        end_frame_in = min(start_frame_in + self.max_frames_input, len(labels_in))

        if load_target:
            # Select the according frames in the output data.
            start_frame_out = start_frame_in * self.len_in_out_multiplier
            # Check if out data is shorter than required.
            end_frame_out = min(end_frame_in * self.len_in_out_multiplier, len(labels_out))
            end_frame_in = end_frame_out // self.len_in_out_multiplier
            end_frame_out = end_frame_in * self.len_in_out_multiplier
            # Trim output data.
            labels_out = self.LabelGenOut.trim_end_sample(labels_out, len(labels_out) - end_frame_out)
            labels_out = self.LabelGenOut.trim_end_sample(labels_out, start_frame_out, reverse=True)

        # Trim input data.
        labels_in = self.LabelGenIn.trim_end_sample(labels_in, len(labels_in) - end_frame_in)
        labels_in = self.LabelGenIn.trim_end_sample(labels_in, start_frame_in, reverse=True)

        # Trim extra data.  # TODO: Untested.
        output_labels_extra = list()
        for label_extra, generator in zip(labels_extra, self.LabelGensExtra):
            raise NotImplementedError("Trim to input or output?")
            label_extra = generator.trim_end_sample(label_extra, len(label_extra) - end_frame_in)
            label_extra = generator.trim_end_sample(label_extra, start_frame_in, reverse=True)
            output_labels_extra.append(label_extra)

        return (labels_in, labels_out, *output_labels_extra)

    def getitem_no_length_check(self, id_name, load_target):
        """Load labels without any length checks, adds embedding indices if given in hparams in constructor."""

        labels_in = self.LabelGenIn[id_name]

        # Process the embedding generation functions.
        if self.f_get_emb_index is not None:
            for f_emb in self.f_get_emb_index:
                labels_in = np.concatenate((labels_in, f_emb(id_name, len(labels_in)).astype(labels_in.dtype)), axis=1)

        labels_out = None
        if load_target:
            labels_out = self.LabelGenOut[id_name]

        labels_extra = list()
        for generator in self.LabelGensExtra:
            labels_extra.append(generator[id_name])

        return (labels_in, labels_out, *labels_extra)

    def get_input(self, id_name):
        return self.LabelGenIn[id_name]

    def get_output(self, id_name):
        return self.LabelGenOut[id_name]

    def postprocess_sample(self, sample, norm_params=None):
        return self.LabelGenOut.postprocess_sample(sample, norm_params)

    def get_dims(self):
        """Returns the feature dimensions of the input and output labels."""
        labels_in, labels_out, *labels_extra = self.__getitem__(0)
        return labels_in.shape[1:], labels_out.shape[1:]  # First dimension is time.
        # return (labels_in.shape[1:], labels_out.shape[1:], *map(lambda x: x.shape[1:], labels_extra))  # First dimension it time.
