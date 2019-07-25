#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#


import unittest

import torch
import numpy
import os

from idiaptts.src.model_trainers.ModelTrainer import ModelTrainer


class TestModelTrainer(unittest.TestCase):

    @staticmethod
    def _get_id_list():
        with open(os.path.join("integration", "fixtures", "database", "file_id_list.txt")) as f:
            id_list = f.readlines()
        # Trim entries in-place.
        id_list[:] = [s.strip(' \t\n\r') for s in id_list]
        return id_list

    def test_split_return_values_torch(self):
        seq_length_output = numpy.array([10, 5])
        output = torch.ones(seq_length_output.max(), 2, 4)

        with unittest.mock.patch.object(ModelTrainer.logger, "error") as mock_logger:
            with self.assertRaises(TypeError):
                ModelTrainer._split_return_values(output, seq_length_output, None, False)
                mock_logger.assert_called_with("No best model exists yet. Continue with the current one.")

    def test_split_return_values(self):
        seq_length_output = numpy.array([10, 6, 8])
        batch_size = 3
        feature_dim = 50
        output = numpy.empty((seq_length_output.max(), batch_size, feature_dim))
        hidden1 = numpy.empty((seq_length_output.max(), batch_size, 2))
        hidden2 = numpy.empty((seq_length_output.max(), batch_size, 4))
        for idx in range(batch_size):
            output[:, idx] = idx
            hidden1[:, idx] = idx * 10
            hidden2[:, idx] = idx * 100
        hidden = (hidden1, hidden2)
        batch = (output, hidden)

        split_batch = ModelTrainer._split_return_values(batch, seq_length_output, None, False)

        for idx in range(batch_size):
            b = split_batch[idx]
            out = b[0]
            h = b[1]
            h1 = h[0]
            h2 = h[1]

            self.assertTrue((out == idx).all(), msg="Output of batch {} is wrong, expected was all values being {}.".format(idx, idx))
            self.assertTrue((h1 == idx * 10).all(), msg="Hidden1 of batch {} is wrong, expected was all values being {}.".format(idx, idx * 10))
            self.assertTrue((h2 == idx * 100).all(), msg="Hidden2 of batch {} is wrong, expected was all values being {}.".format(idx, idx * 100))

    def test_split_return_values_batch_first(self):
        seq_length_output = numpy.array([10, 6, 8])
        batch_size = 3
        feature_dim = 50
        output = numpy.empty((batch_size, seq_length_output.max(), feature_dim))
        hidden1 = numpy.empty((batch_size, seq_length_output.max(), 2))
        hidden2 = numpy.empty((batch_size, seq_length_output.max(), 4))
        for idx in range(batch_size):
            output[idx] = idx
            hidden1[idx] = idx * 10
            hidden2[idx] = idx * 100
        hidden = (hidden1, hidden2)
        batch = (output, hidden)

        split_batch = ModelTrainer._split_return_values(batch, seq_length_output, None, True)

        for idx in range(batch_size):
            b = split_batch[idx]
            out = b[0]
            h = b[1]
            h1 = h[0]
            h2 = h[1]

            self.assertTrue((out == idx).all(),
                            msg="Output of batch {} is wrong, expected was all values being {}.".format(idx, idx))
            self.assertTrue((h1 == idx * 10).all(),
                            msg="Hidden1 of batch {} is wrong, expected was all values being {}.".format(idx, idx * 10))
            self.assertTrue((h2 == idx * 100).all(),
                            msg="Hidden2 of batch {} is wrong, expected was all values being {}.".format(idx, idx * 100))

    def test_input_to_str_list(self):
        # Tuple input but elements are not strings.
        out = ModelTrainer._input_to_str_list((121, 122))
        self.assertEqual(["121", "122"], out)

        # Valid path to file id list.
        out = ModelTrainer._input_to_str_list(os.path.join("integration", "fixtures", "file_id_list.txt"))
        self.assertEqual(TestModelTrainer._get_id_list(), out)

        # Single input id.
        out = ModelTrainer._input_to_str_list("121")
        self.assertEqual("121", out)

        # Wrong input.
        with self.assertRaises(ValueError):
            ModelTrainer._input_to_str_list(numpy.array([1, 2]))
