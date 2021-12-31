#
# Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#


import unittest

import os
from typing import List

from torch.utils.data import DataLoader

from idiaptts.src.data_preparation.DataReaderConfig import DataReaderConfig
from idiaptts.src.neural_networks.pytorch.ModularModelHandlerPyTorch import ModularModelHandlerPyTorch
from idiaptts.src.data_preparation.PyTorchDatareadersDataset import PyTorchDatareadersDataset


class TestPyTorchDatareadersDataset(unittest.TestCase):

    @staticmethod
    def _get_datareader(id_list: List[str],
                        data_reader_configs: List[DataReaderConfig]):

        datareaders = [c.create_reader() for c in data_reader_configs]
        return PyTorchDatareadersDataset(id_list, datareaders)

    @staticmethod
    def _get_id_list():
        with open(os.path.join("integration", "fixtures", "database",
                               "file_id_list.txt")) as f:
            id_list = f.readlines()
        # Trim entries in-place.
        id_list[:] = [s.strip(' \t\n\r') for s in id_list]
        return id_list

    def _get_datareader_configs(self):

        dir_world_features = "integration/fixtures/WORLD"
        dir_question_labels = "integration/fixtures/questions"
        num_coded_sps = 20
        n_frames_per_step = 1

        datareader_configs = [
            DataReaderConfig(
                name="cmp_features",
                feature_type="WorldFeatLabelGen",
                directory=dir_world_features,
                features=["cmp_mcep" + str(num_coded_sps)],
                output_names=["acoustic_features"],
                chunk_size=n_frames_per_step,
                requires_seq_mask=True,
                num_coded_sps=num_coded_sps,
                add_deltas=True,
                match_length="questions"
            ),
            DataReaderConfig(
                name="questions",
                feature_type="QuestionLabelGen",
                directory=dir_question_labels,
                features="questions",
                chunk_size=n_frames_per_step,
                num_questions=409,
                match_length="cmp_features"
            )
        ]

        return datareader_configs

    def test_match_length_cycle(self):

        datareader_configs = self._get_datareader_configs()
        id_list = self._get_id_list()
        datareader = self._get_datareader(
            id_list=id_list, data_reader_configs=datareader_configs)

        dataloader = DataLoader(
            dataset=datareader,
            collate_fn=ModularModelHandlerPyTorch.prepare_batch)

        next(iter(dataloader))
        next(iter(dataloader))

    def test_match_length_2d(self):
        dir_dur_labels = "integration/fixtures/dur"
        datareader_configs = self._get_datareader_configs()
        datareader_configs.append(
            DataReaderConfig(
                name="durations",
                feature_type="PhonemeDurationLabelGen",
                directory=dir_dur_labels,
                features="durations",
                chunk_size=datareader_configs[0].chunk_size,
                load_as_matrix=True,
                output_names=["attention_matrix"],
                match_length=["cmp_features", "questions"]
            )
        )

        id_list = self._get_id_list()
        datareader = self._get_datareader(
            id_list=id_list, data_reader_configs=datareader_configs)

        dataloader = DataLoader(
            dataset=datareader,
            collate_fn=ModularModelHandlerPyTorch.prepare_batch)

        next(iter(dataloader))
        next(iter(dataloader))
