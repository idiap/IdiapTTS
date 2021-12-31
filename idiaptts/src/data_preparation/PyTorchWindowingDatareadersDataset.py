#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#

# System imports.
import logging
import queue
import random
from typing import List

# Third-party imports.
import numpy as np
import torch
import torch.multiprocessing
from torch.utils.data.dataset import IterableDataset

# Local imports.
from idiaptts.src.data_preparation.PyTorchDatareadersDataset import PyTorchDatareadersDataset
from idiaptts.src.ExtendedHParams import ExtendedHParams


class PyTorchWindowingDatareadersDataset(IterableDataset):

    logger = logging.getLogger(__name__)
    _id_list = torch.multiprocessing.Queue()
    queue_filled = torch.multiprocessing.Event()
    queue_filled_timeout_sec = 60

    def __init__(self, id_list: List[str], datareaders: List,
                 hparams: ExtendedHParams, is_train_set: bool = False,
                 is_val_set: bool = False, is_test_set: bool = False):
        super().__init__()

        assert hparams.has_value("windowed_feature_names"), \
            "Use hparams.windowed_feature_names to define the features to " \
            "apply the windowing to. Those features have to match in length."

        self.id_list = id_list
        if is_train_set:
            self.batch_size = hparams.batch_size_train
        elif is_val_set:
            self.batch_size = hparams.batch_size_val
        elif is_test_set:
            self.batch_size = hparams.batch_size_test

        self.windowed_feature_names = hparams.windowed_feature_names
        self.window_size = hparams.get("window_size", 500)
        assert self.window_size > 1
        self.step_size = hparams.get("step_size", 50)
        self.mem_copy = hparams.get("windower_mem_copy", False)
        self.allow_shorter_sequences = hparams.get("allow_shorter_sequences",
                                                   True)
        self.random_offset = hparams.get("windower_random_offset", True)

        self.dataset = PyTorchDatareadersDataset(id_list, datareaders, hparams)
        self.length = None

    def __len__(self):
        if self.length is None:
            self.length = self._get_total_length()
        return self.length

    def _get_total_length(self):
        self.logger.info("Estimating datareader length. This can take some time.")
        reader = self.dataset.get_datareader_by_output_name(
            self.windowed_feature_names[0])
        total_num_chunks = 0
        for id_name in self.id_list:
            feature_len = reader.get_length(id_name)
            num_chunks = self._length_to_num_chunks(feature_len)
            total_num_chunks += num_chunks

        return total_num_chunks // self.batch_size

    def _length_to_num_chunks(self, length):
        if length < self.window_size:
            if self.allow_shorter_sequences:
                num_chunks = 1
            else:
                num_chunks = 0
        elif length == self.window_size:
            num_chunks = 1
        else:
            num_chunks = (length - self.window_size) // self.step_size + 1
            if self.allow_shorter_sequences:
                num_chunks += 1

        return num_chunks

    def process_data(self, batch_idx):
        # worker = torch.utils.data.get_worker_info()
        # worker_id = worker.id if worker is not None else -1
        while True:
            try:
                # print(worker_id, "queue size:", PyTorchWindowingDatareadersDataset._id_list.qsize())
                id_name = PyTorchWindowingDatareadersDataset._id_list.get(block=False)
            except queue.Empty:
                return

            output_dict = self.dataset.get_id_name(id_name)[0]

            feature_len = len(output_dict[self.windowed_feature_names[0]])
            if self.random_offset:
                offset = random.randrange(self.step_size)
                feature_len -= offset
            else:
                offset = 0
            num_chunks = self._length_to_num_chunks(feature_len)

            for chunk_idx in random.sample(range(num_chunks), num_chunks):
            # for chunk_idx in range(num_chunks):
                win_output_dict = output_dict.copy()
                start_idx = 0
                end_idx = -1
                len_feature = -1
                for feature_name in self.windowed_feature_names:
                    feature = output_dict[feature_name]

                    len_feature = len(feature)
                    if len_feature > self.window_size:
                        start_idx = offset + chunk_idx * self.step_size
                        end_idx = start_idx + self.window_size

                        win_features = feature[start_idx:end_idx]

                        if self.mem_copy:
                            win_features = win_features.copy()
                        win_output_dict[feature_name] = win_features

                # win_output_dict["_test"] = np.array((start_idx, end_idx, len_feature))
                # win_output_dict["_worker"] = np.array(worker_id)[None]
                yield win_output_dict, self.dataset

    def get_streams(self):
        return zip(*[self.process_data(idx) for idx in range(self.batch_size)])

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:  # num_workers == 0
            PyTorchWindowingDatareadersDataset._fill_queue(self.id_list)
        else:
            queue_filled = PyTorchWindowingDatareadersDataset.queue_filled.wait(
                PyTorchWindowingDatareadersDataset.queue_filled_timeout_sec)

            if not queue_filled:
                raise RuntimeError("Timeout on filling the queue.")

        # print("Queue filled ", worker_info.id)
        return self.get_streams()

    @staticmethod
    def worker_init_fn(worker_id):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None or worker_info.id == 0:
            PyTorchWindowingDatareadersDataset._fill_queue(
                worker_info.dataset.id_list)

    @staticmethod
    def _fill_queue(id_list):
        for id_name in id_list:
            PyTorchWindowingDatareadersDataset._id_list.put(id_name)
        PyTorchWindowingDatareadersDataset.queue_filled.set()
