#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#

# System imports.
from typing import Union, Any, List, Dict, Optional, cast, Set
import logging

# Third-party imports.
import numpy as np
import torch
from torch.utils.data import Dataset

from idiaptts.src.ExtendedHParams import ExtendedHParams


class PyTorchDatareadersDataset(Dataset):

    def __init__(self, id_list: List[str], datareaders: List, *args, **kwargs):
        self.id_list = id_list

        self.datareaders = set(datareaders)

    def get_input_dim(self, input_names=None):
        """Returns the feature dimensions of the input and output labels."""
        output_dict, _ = self.__getitem__(0)
        dims = [output_dict[name].shape[1] for name in input_names]  # First dimension it time.
        return sum(dims)
        # return (labels_in.shape[1:], labels_out.shape[1:], *map(lambda x: x.shape[1:], labels_extra))  # First dimension it time.

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, item):
        id_name = self.id_list[item]
        return self.get_id_name(id_name)

    def get_id_name(self, id_name):
        output_dict = {}
        for reader in self.datareaders:
            # NOTE: Do not change to reader[id_name], because this does not know
            #       about the updated __getitem__ from DataReaderConfig.
            reader_output = reader.__getitem__(id_name)

            for key in reader_output.keys():
                if key != "_id_list" and key in output_dict:
                    raise KeyError("Feature {} defined twice.".format(key))

            output_dict.update(reader_output)

        self._match_output_lengths(output_dict, id_name)

        self._match_max_frames(output_dict, id_name)

        # self._match_min_frames(output_dict, id_name)

        return output_dict, self

    def _get_datareader_output(self, reader, id_name, ref_length=None):

        reader_output_dict = reader.__getitem__(id_name)

        # if reader.random_select:
        #     for key, value in reader_output_dict:
        #         len_value = len(value)
        #         start_frame = torch.IntTensor(1).random_(0, max(1, len_value - reader.max_frames))
        #         # Check if data is shorter than max_frames_input.
        #         end_frame = min(start_frame + reader.max_frames_input, len_value)
        #         reader_output_dict[key] = reader.trim(value, start_frame, len_value - end_frame)

        return reader_output_dict

        if reader.match_length is not None:
            ref_lengths = self._get_ref_lengths(reader.match_length, id_name)

            for key, value in reader_output_dict.items():
                if key == "_id_list":
                    continue
                len_diff = len(value) - ref_length
                if len_diff > 0:
                    trim_front = len_diff // 2
                    trim_end = len_diff - trim_front
                    self._trim_datareader_output(reader, ref_lengths, value)
                # if len_diff > 0:
                #     trim_front = len_diff // 2
                #     trim_end = len_diff - trim_front
                #     reader_output_dict[key] = reader.trim(value, trim_front, trim_end)
                # elif len_diff < 0:
                #     pad_front = abs(len_diff) // 2
                #     pad_end = abs(len_diff) - pad_front
                #     reader_output_dict[key] = reader.pad(value, pad_front, pad_end)
                reader_output_dict[key] = self._pad_datareader_output(reader, ref_lengths, value)

        return reader_output_dict

    def _match_output_lengths(self, output_dict, id_name):
        was_trimmed = True
        known_length = {}
        while was_trimmed:
            was_trimmed = False
            for reader in self.datareaders:
                if reader.match_length is not None:
                    was_trimmed = self._trim_datareader(reader, output_dict,
                                                        id_name, known_length)
                    if was_trimmed:
                        break

    def _trim_datareader(self, reader, output_dict, id_name, known_length):
        was_trimmed = False

        ref_lengths = self._get_ref_lengths(reader.match_length,
                                            id_name,
                                            known_length)

        for key in reader.output_names:
            if key == "_id_list":
                continue
            value = output_dict[key]
            try:
                output_dict[key], was_trimmed = self._trim_datareader_output(
                    reader, ref_lengths, value)

                if was_trimmed:
                    known_length[reader.name] = ref_lengths
                    logging.debug("Trimmed {} ({}) to {} ({}).".format(
                        key, value.shape, reader.match_length, ref_lengths))
                    break
            except ValueError:
                logging.debug("Cannot trim {} ({}) to longer reference feature "
                              "{} ({}). Trim reference instead." .format(
                                  key, value.shape, reader.match_length,
                                  ref_lengths))

        return was_trimmed

    def _get_ref_lengths(self, match_length: List[str], id_name: str,
                         known_length: Dict[str, int]):
        ref_lengths = []

        for name in match_length:
            reader = self.get_datareader_by_name(name)
            if reader.name not in known_length:
                # Save as list in known_length for 2D matches
                known_length[reader.name] = [reader.get_length(id_name)]
            ref_lengths.append(known_length[reader.name][0])

        return ref_lengths

    def get_datareader_by_name(self, name):
        for reader in self.datareaders:
            if name == reader.name:
                return reader
        raise KeyError("No data reader named {} found in {}.".format(name, [r.name for r in self.datareaders]))

    def get_datareader_by_output_name(self, name):
        for reader in self.datareaders:
            if name in reader.output_names:
                return reader
        raise KeyError("No data reader with output name {} found in {}.".format(name, [r.name for r in self.datareaders]))

    # def _pad_datareader_output(self, reader, ref_lengths, value):
    #     pad_width = []
    #     for dim, ref_length in enumerate(ref_lengths):
    #         len_diff = ref_length - value.shape[dim]
    #         front = len_diff // 2
    #         end = len_diff - front
    #         pad_width.append((front, end))

    #         if len_diff < 0:
    #             raise NotImplementedError()
    #             # return reader.trim(value, front, end)
    #     pad_width += [(0, 0)] * (value.ndim - len(ref_lengths))

    #     return reader.pad(value, pad_width)

    def _trim_datareader_output(self, reader, ref_lengths, value):
        trim_width = []
        do_trimming = False
        for dim, ref_length in enumerate(ref_lengths):
            len_diff = ref_length - value.shape[dim]
            if len_diff > 0:
                raise ValueError()

            front = (-len_diff) // 2
            end = -len_diff - front
            trim_width.append((front, end))
            do_trimming |= front != 0 or end != 0

        # trim_width += [None] * (value.ndim - len(ref_lengths))

        if do_trimming:
            return reader.trim(value, trim_width), True
        else:
            return value, False

    def _match_max_frames(self, output_dict, id_name):
        """
        For datareader with the max_frames variable set, select a subset of
        max_frames and do the same for readers named in match_length that also
        have the max_frames variable set. The search for match_length readers
        continues recursively to guarantee equal length with chained
        match_length dependencies. It can also handles cyclic dependencies.
        """

        # Maintain a set of processed readers to break possible cyclic
        # match_length dependencies.
        processed_readers = set()
        for reader in self.datareaders:
            if reader.max_frames is None:
                continue

            feature_len = reader.get_length(id_name)
            if feature_len <= reader.max_frames:
                return
            if reader.random_select:
                start_frame = torch.IntTensor(1).random_(
                    0, max(1, feature_len - reader.max_frames))
            else:
                start_frame = 0

            end_frame = min(start_frame + reader.max_frames, feature_len)

            self._select_max_frames(processed_readers, output_dict, reader,
                                    start_frame, end_frame)

    def _select_max_frames(self,
                           processed_readers: Set,
                           output_dict: Dict[str, object],
                           reader,
                           start_frame: int,
                           end_frame: int):
        processed_readers.add(reader)
        for name in reader.output_names:
            logging.debug("Select frames {} to {} in {}.".format(
                start_frame, end_frame, name))
            output_dict[name] = output_dict[name][start_frame:end_frame]
        if reader.match_length is None:
            return
        for reader_name in reader.match_length:
            reader = self.get_datareader_by_name(reader_name)
            if reader.max_frames is not None and reader not in processed_readers:
                self._select_max_frames(processed_readers, output_dict, reader,
                                        start_frame, end_frame)
