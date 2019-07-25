#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#


import unittest

import os
import shutil
import filecmp
import torch
import logging
import numpy

from idiaptts.src.neural_networks.pytorch.ModelHandlerPyTorch import ModelHandlerPyTorch
from idiaptts.src.model_trainers.ModelTrainer import ModelTrainer
# from idiaptts.src.data_preparation.questions.QuestionLabelGen import QuestionLabelGen
# from idiaptts.src.data_preparation.world.WorldFeatLabelGen import WorldFeatLabelGen
# from idiaptts.src.data_preparation.PyTorchLabelGensDataset import PyTorchLabelGensDataset as LabelGensDataset
from idiaptts.misc.utils import makedirs_safe
from idiaptts.src.neural_networks.pytorch.utils import equal_checkpoint


class TestModelHandlerPyTorch(unittest.TestCase):

    out_dir = None

    @classmethod
    def setUpClass(cls):
        cls.out_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), type(cls()).__name__)
        makedirs_safe(cls.out_dir)  # Create class name directory.

    @classmethod
    def tearDownClass(cls):
        os.rmdir(cls.out_dir)  # Remove class name directory, should be empty.

    def test_save_load_equality(self):
        hparams = ModelTrainer.create_hparams()
        hparams.out_dir = os.path.join(self.out_dir, "test_save_load_equality")  # Add function name to path.
        model_path = os.path.join(hparams.out_dir, "test_model.nn")

        # Create a new model and save it.
        dim_in, dim_out = 10, 4
        model_handler = ModelHandlerPyTorch(hparams)
        model_handler.model = torch.nn.Sequential(torch.nn.Linear(dim_in, dim_out))
        model_handler.save_model(model_path)

        # Create a new model handler and test load save.
        hparams.model_type = None
        model_handler = ModelHandlerPyTorch(hparams)
        model_handler.load_model(model_path, False)
        model_copy_path = os.path.join(hparams.out_dir, "test_model_copy.nn")
        model_handler.save_model(model_copy_path)

        # self.assertTrue(filecmp.cmp(model_path, model_copy_path, False))  # This does not work.
        self.assertTrue(equal_checkpoint(model_path, model_copy_path), "Loaded and saved models are not the same.")

        shutil.rmtree(hparams.out_dir)
