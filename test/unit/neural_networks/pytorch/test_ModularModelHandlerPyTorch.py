#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#


import unittest

import os
import shutil
import torch

from idiaptts.misc.utils import makedirs_safe
from idiaptts.src.model_trainers.ModularTrainer import ModularTrainer
import idiaptts.src.neural_networks.pytorch.models.rnn_dyn as rnn_dyn
from idiaptts.src.neural_networks.pytorch.ModularModelHandlerPyTorch import ModularModelHandlerPyTorch
from idiaptts.src.neural_networks.pytorch.utils import equal_iterable


class TestModularHandlerPyTorch(unittest.TestCase):

    out_dir = None

    @classmethod
    def setUpClass(cls):
        cls.out_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                   type(cls()).__name__)
        makedirs_safe(cls.out_dir)  # Create class name directory.

    @classmethod
    def tearDownClass(cls):
        os.rmdir(cls.out_dir)  # Remove class name directory, should be empty.

    def test_save_load_equality(self):
        hparams = ModularTrainer.create_hparams()
        hparams.optimiser_type = "Adam"
        hparams.optimiser_args["lr"] = 0.1
        # Add function name to path.
        out_dir = os.path.join(self.out_dir, "test_save_load_equality")
        model_path = os.path.join(out_dir, "test_model")

        # Create a new model, run the optimiser once to obtain a state, and save everything.
        in_dim, out_dim = 10, 4
        total_epochs = 10
        model_handler = ModularModelHandlerPyTorch()
        model_handler.model = rnn_dyn.Config(in_dim=in_dim, layer_configs=[
            rnn_dyn.Config.LayerConfig(layer_type="Linear", out_dim=out_dim)
        ]).create_model()
        model_handler.set_optimiser(hparams)

        seq_length = torch.tensor((10, 7), dtype=torch.long)
        batch_size = 2
        test_input = torch.ones([seq_length[0], batch_size, in_dim])
        model_handler.model.init_hidden(batch_size)
        output = model_handler.model(test_input, seq_lengths_input=seq_length,
                                     max_length_inputs=seq_length.max())[0]
        output.mean().backward()

        model_handler.optimiser.step()
        model_handler.save_checkpoint(epoch=total_epochs, model_path=model_path)

        # Create a new model handler and test load save.
        model_handler_copy = ModularModelHandlerPyTorch()
        model_handler_copy.load_checkpoint(
            hparams,
            model_path=model_path,
            load_optimiser=True,
            epoch=total_epochs,
            verbose=False)

        zip_params = zip(model_handler.model.parameters(),
                         model_handler_copy.model.parameters())
        self.assertTrue(all([(x == x_copy).all() for x, x_copy in zip_params]),
                        "Loaded and saved models are not the same.")
        current_opt_state = model_handler.optimiser.state_dict()["state"]
        copy_opt_state = model_handler_copy.optimiser.state_dict()["state"]
        self.assertTrue(equal_iterable(current_opt_state, copy_opt_state),
                        "Loaded and saved optimisers are not the same.")

        shutil.rmtree(out_dir)
