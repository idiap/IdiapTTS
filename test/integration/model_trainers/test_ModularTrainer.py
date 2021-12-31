#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#


import unittest
from functools import partial
from unittest.mock import Mock

import os
import shutil
import filecmp
import torch
import soundfile
import pydub
import array
import logging
import itertools
import numpy
import copy
from distutils.dir_util import copy_tree

from idiaptts.src.model_trainers.ModularTrainer import ModularTrainer
from idiaptts.src.neural_networks.pytorch.utils import equal_iterable
from idiaptts.misc.utils import makedirs_safe
from idiaptts.src.data_preparation.DataReaderConfig import DataReaderConfig
from idiaptts.src.neural_networks.pytorch.loss.NamedLoss import NamedLoss
from idiaptts.src.neural_networks.pytorch.loss.VAEKLDLoss import VAEKLDLoss
import idiaptts.src.neural_networks.pytorch.models.rnn_dyn as rnn_dyn
from idiaptts.src.neural_networks.pytorch.models.NamedForwardWrapper import NamedForwardWrapper


class TestModularTrainer(unittest.TestCase):

    post_processing_mapping = {"pred_acoustic_features": "cmp_features"}

    @classmethod
    def setUpClass(cls):
        hparams = cls._get_hparams(cls())
        makedirs_safe(hparams.out_dir)  # Create class name directory.
        # Load test data
        cls.id_list = cls._get_id_list()

    @classmethod
    def tearDownClass(cls):
        hparams = cls._get_hparams(cls())
        os.rmdir(hparams.out_dir)  # Remove class name directory, should be empty.

    def _get_hparams(self):
        hparams = ModularTrainer.create_hparams()
        # General parameters
        hparams.add_hparam("num_questions", 409)
        hparams.epochs = 0
        hparams.test_set_perc = 0.05
        hparams.val_set_perc = 0.05
        hparams.optimiser_args["lr"] = 0.02
        hparams.seed = None  # Remove the default seed.
        hparams.out_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), type(self).__name__)
        hparams.num_coded_sps = 20
        hparams.add_hparam("n_frames_per_step", 7)
        # hparams.add_hparam("add_deltas", True)

        # Training parameters.
        hparams.epochs = 0
        hparams.model_name = "test_model"
        hparams.scheduler_type = "None"

        return hparams

    @staticmethod
    def _get_id_list():
        with open(os.path.join("integration", "fixtures", "database", "file_id_list.txt")) as f:
            id_list = f.readlines()
        # Trim entries in-place.
        id_list[:] = [s.strip(' \t\n\r') for s in id_list]
        return id_list

    def _get_trainer(self, hparams, model_config=None):
        # loss_configs = self._get_loss_configs()

        trainer = ModularTrainer(id_list=self.id_list, hparams=hparams)

        return trainer

    def _get_data_reader_configs(self, hparams):
        dir_world_features = "integration/fixtures/WORLD"
        dir_question_labels = "integration/fixtures/questions"

        datareader_configs = [
            DataReaderConfig(
                name="cmp_features",
                feature_type="WorldFeatLabelGen",
                directory=dir_world_features,
                features=["cmp_mcep" + str(hparams.num_coded_sps)],
                output_names=["acoustic_features"],
                chunk_size=hparams.n_frames_per_step,
                requires_seq_mask=True,
                num_coded_sps=hparams.num_coded_sps,
                add_deltas=True,
                match_length="questions"
            ),
            DataReaderConfig(
                name="questions",
                feature_type="QuestionLabelGen",
                directory=dir_question_labels,
                features="questions",
                chunk_size=hparams.n_frames_per_step,
                num_questions=hparams.num_questions,
                match_length="cmp_features"
            )
        ]

        return datareader_configs

    def _get_loss_configs(self):
        return [
            NamedLoss.Config(
                name="MSE_acoustic_features",
                type_="MSELoss",
                input_names=["pred_acoustic_features", "acoustic_features"],
                seq_mask="acoustic_features_mask"
            )
        ]

    def test_init_hparams_none(self):
        hparams = self._get_hparams()

        data_reader_configs = self._get_data_reader_configs(hparams)

        with self.assertRaises(AssertionError):
            ModularTrainer(id_list=self.id_list, data_reader_configs=data_reader_configs, hparams=None)

    def test_init_no_model_type_and_saved_model(self):
        hparams = self._get_hparams()
        hparams.out_dir = os.path.join(hparams.out_dir, "test_init_no_model_type_and_saved_model")  # Add function name to path.
        hparams.load_checkpoint_epoch = -1
        trainer = self._get_trainer(hparams)
        loss_configs = self._get_loss_configs()
        # Check fail when neither model_type is given nor a model exists with model_name.
        with unittest.mock.patch.object(trainer.logger, "error") as mock_logger:
            file_name = os.path.join(hparams.out_dir, hparams.model_name, hparams.networks_dir)
            self.assertRaises(FileNotFoundError, trainer.init, hparams, data_reader_configs=self._get_data_reader_configs(hparams))
            mock_logger.assert_called_with("Model does not exist at {}. [Errno 2] No such file or directory: '{}'".format(file_name, os.path.join(file_name, "params_best")))
        shutil.rmtree(hparams.out_dir)

    def test_init_test55_val55_set_split(self):
        hparams = self._get_hparams()

        with self.assertRaises(AssertionError):
            hparams.val_set_perc = .55
            hparams.test_set_perc = .55
            self._get_trainer(hparams)

    def test_init_test0_val0_set_split(self):
        hparams = self._get_hparams()
        hparams.val_set_perc = 0.0
        hparams.test_set_perc = 0.0
        trainer = self._get_trainer(hparams)

        self.assertEqual(len(self.id_list), len(trainer.id_list_train))
        self.assertIsNone(trainer.id_list_val)
        self.assertIsNone(trainer.id_list_test)

    def test_init_test33_val0_set_split(self):
        hparams = self._get_hparams()
        hparams.val_set_perc = 0.0
        hparams.test_set_perc = 0.33
        trainer = self._get_trainer(hparams)

        self.assertIsNone(trainer.id_list_val)

        expected_length = max(1, int(len(self.id_list) * hparams.test_set_perc))
        self.assertEqual(expected_length, len(trainer.id_list_test))
        self.assertEqual(len(self.id_list) - expected_length, len(trainer.id_list_train))

    def test_init_test0_val66_set_split(self):
        hparams = self._get_hparams()
        hparams.val_set_perc = 0.66
        hparams.test_set_perc = 0.0
        trainer = self._get_trainer(hparams)

        self.assertIsNone(trainer.id_list_test)

        expected_length = max(1, int(len(self.id_list) * hparams.val_set_perc))
        self.assertEqual(expected_length, len(trainer.id_list_val))
        self.assertEqual(len(self.id_list) - expected_length, len(trainer.id_list_train))

    def test_init_test20_val40_set_split(self):
        hparams = self._get_hparams()
        hparams.val_set_perc = 0.4
        hparams.test_set_perc = 0.2
        trainer = self._get_trainer(hparams)

        # Check length of id lists.
        expected_val_length = max(1, int(len(self.id_list) * hparams.val_set_perc))
        expected_test_length = max(1, int(len(self.id_list) * hparams.test_set_perc))
        self.assertEqual(expected_test_length, len(trainer.id_list_test))
        self.assertEqual(expected_val_length, len(trainer.id_list_val))
        self.assertEqual(len(self.id_list) - expected_test_length - expected_val_length, len(trainer.id_list_train))
        self.assertEqual(len(self.id_list), len(trainer.id_list_train) + len(trainer.id_list_test) + len(trainer.id_list_val))

        # Check that all lists are disjoint.
        test_in_train = list(set(trainer.id_list_train).intersection(trainer.id_list_test))
        self.assertEqual(0, len(test_in_train), msg="Found test id(s) in train set: {}".format(test_in_train))
        val_in_train = list(set(trainer.id_list_train).intersection(trainer.id_list_val))
        self.assertEqual(0, len(val_in_train), msg="Found validation id(s) in train set: {}".format(val_in_train))
        val_in_test = list(set(trainer.id_list_val).intersection(trainer.id_list_test))
        self.assertEqual(0, len(val_in_test), msg="Found validation id(s) in test set: {}".format(val_in_test))

    def test_init_e0_create(self):
        # Try epochs=0, loading will fail because no model exists, create a new one.
        hparams = self._get_hparams()
        hparams.epochs = 0
        hparams.out_dir = os.path.join(hparams.out_dir, "test_init_e0_create")  # Add function name to path.
        hparams.model_type = "RNNDYN-1_RELU_32-1_FC_67"
        model_config = NamedForwardWrapper.Config(
            wrapped_model_config=rnn_dyn.convert_legacy_to_config(in_dim=hparams.num_questions, hparams=hparams),
            input_names=['questions'],
            output_names=['pred_acoustic_features'],
            batch_first=hparams.batch_first)
        # loss_configs = self._get_loss_configs()
        trainer = self._get_trainer(hparams)
        trainer.init(hparams, model_config, data_reader_configs=self._get_data_reader_configs(hparams))

        # Check if model is loaded but not saved and no checkpoint exists.
        self.assertIsNotNone(trainer.model_handler.model)
        self.assertFalse(os.path.isfile(os.path.join(hparams.out_dir, hparams.model_name, hparams.networks_dir,
                                                     "config.json")))
        self.assertEqual(0, len([name for name in os.listdir(os.path.join(
            hparams.out_dir,
            hparams.model_name,
            hparams.networks_dir)) if os.path.isfile(name)]))

        shutil.rmtree(hparams.out_dir)

    def test_init_e0_load(self):
        # Try epochs=0, loading existing model.
        hparams = self._get_hparams()
        hparams.use_gpu = True
        hparams.epochs = 0
        hparams.out_dir = os.path.join(hparams.out_dir, "test_init_e0_load")  # Add function name to path.
        hparams.model_type = None
        hparams.load_checkpoint_epoch = 0
        hparams.load_optimiser = False

        with unittest.mock.patch.object(ModularTrainer.logger, "warning") as mock_logger:
            trainer = self._get_trainer(hparams)
            mock_logger.assert_called_with("No CUDA device available, use CPU mode instead.")

        target_dir = os.path.join(hparams.out_dir, hparams.model_name)
        makedirs_safe(target_dir)
        copy_tree(os.path.join("integration", "fixtures", "test_model_in409_out67"), target_dir, verbose=True)
        trainer.init(hparams, data_reader_configs=self._get_data_reader_configs(hparams))
        self.assertIsNotNone(trainer.model_handler.model)

        shutil.rmtree(hparams.out_dir, ignore_errors=True)

    def test_ignore_layer(self):
        hparams = self._get_hparams()
        hparams.use_gpu = False
        hparams.epochs = 0
        hparams.out_dir = os.path.join(hparams.out_dir, "test_ignore_layer")  # Add function name to path.
        hparams.model_type = None
        hparams.load_checkpoint_epoch = 0
        hparams.load_optimiser = False

        trainer = self._get_trainer(hparams)

        target_dir = os.path.join(hparams.out_dir, hparams.model_name)
        makedirs_safe(target_dir)
        copy_tree(os.path.join("integration", "fixtures", "test_model_in409_out67"), target_dir)
        trainer.init(hparams, data_reader_configs=self._get_data_reader_configs(hparams))
        saved_weights = trainer.model_handler.model.model[0][0].bias.detach().clone()
        saved_weights_2 = trainer.model_handler.model.model[1][0].bias.detach().clone()

        hparams.ignore_layers = ['model.1.module.0.bias', r'model\.2.*']
        trainer = self._get_trainer(hparams)
        trainer.init(hparams, data_reader_configs=self._get_data_reader_configs(hparams))
        self.assertFalse(torch.isclose(saved_weights, trainer.model_handler.model.model[0][0].bias).all())
        self.assertFalse(torch.isclose(saved_weights_2, trainer.model_handler.model.model[1][0].bias).all())

        hparams.ignore_layers = ['wrong_key']
        with unittest.mock.patch.object(trainer.logger, "error") as mock_logger:
            trainer = self._get_trainer(hparams)
            self.assertRaises(KeyError, trainer.init, hparams, data_reader_configs=self._get_data_reader_configs(hparams))

        shutil.rmtree(hparams.out_dir)

    def test_init_create(self):
        # Try epochs=3, creating a new model.
        hparams = self._get_hparams()
        hparams.out_dir = os.path.join(hparams.out_dir, "test_init_create")  # Add function name to path.
        hparams.model_type = "RNNDYN-1_RELU_32-1_FC_67"
        model_config = NamedForwardWrapper.Config(
            wrapped_model_config=rnn_dyn.convert_legacy_to_config(in_dim=hparams.num_questions, hparams=hparams),
            input_names=['questions'],
            output_names=['pred_acoustic_features'],
            batch_first=hparams.batch_first)
        trainer = self._get_trainer(hparams)
        trainer.init(hparams=hparams, model_config=model_config, data_reader_configs=self._get_data_reader_configs(hparams))
        self.assertIsNotNone(trainer.model_handler.model)

        shutil.rmtree(hparams.out_dir)

    def test_init_load(self):
        # Try epochs=3, loading existing model.
        hparams = self._get_hparams()
        hparams.out_dir = os.path.join(hparams.out_dir, "test_init_load")  # Add function name to path.
        hparams.load_checkpoint_step = 4
        target_dir = os.path.join(hparams.out_dir, hparams.model_name)
        makedirs_safe(target_dir)
        copy_tree(os.path.join("integration", "fixtures", "test_model_in409_out67"), target_dir)
        trainer = self._get_trainer(hparams)
        trainer.init(hparams, data_reader_configs=self._get_data_reader_configs(hparams))
        self.assertIsNotNone(trainer.model_handler.model)

        shutil.rmtree(hparams.out_dir)

    def test_train_e0(self):
        hparams = self._get_hparams()
        hparams.out_dir = os.path.join(hparams.out_dir, "test_train_e0")
        hparams.model_type = "RNNDYN-1_RELU_32-1_FC_67"
        trainer = self._get_trainer(hparams)

        model_config = NamedForwardWrapper.Config(
            wrapped_model_config=rnn_dyn.convert_legacy_to_config(
                in_dim=hparams.num_questions, hparams=hparams),
            input_names=['questions'],
            output_names=['pred_acoustic_features'],
            batch_first=hparams.batch_first)
        hparams.load_checkpoint_epoch = 0
        with unittest.mock.patch.object(trainer.logger, "error") as mock_logger:
            file_name = os.path.join(hparams.out_dir, hparams.model_name,
                                     hparams.networks_dir)
            self.assertRaises(
                FileNotFoundError, trainer.init, hparams,
                data_reader_configs=self._get_data_reader_configs(hparams))
            mock_logger.assert_called_with(
                "Model does not exist at {}. [Errno 2] No such file or "
                "directory: '{}'".format(
                    file_name,
                    os.path.join(
                        file_name,
                        "params_e" + str(hparams.load_checkpoint_epoch))))

        hparams.load_checkpoint_epoch = None
        trainer.init(
            hparams, model_config,
            data_reader_configs=self._get_data_reader_configs(hparams))
        all_loss, all_loss_train, _ = trainer.train(hparams)
        # Check if no training happened.
        self.assertEqual(0, len(all_loss))
        self.assertEqual(0, len(all_loss_train))

        shutil.rmtree(hparams.out_dir)

    def test_train_multiloss_and_loss_plot(self):

        loss_configs = [
            NamedLoss.Config(
                name="MSE_acoustic_features",
                type_="MSELoss",
                input_names=["pred_acoustic_features", "acoustic_features"],
                seq_mask=None,
                reduction='mean'
            ),
            NamedLoss.Config(
                name="L1",
                type_="L1Loss",
                input_names=["pred_acoustic_features", "acoustic_features"],
                seq_mask=None,
                reduction='mean'
            )
        ]

        hparams = self._get_hparams()
        hparams.seed = 42
        hparams.out_dir = os.path.join(hparams.out_dir, "test_train_multiloss")  # Add function name to path.
        trainer = self._get_trainer(hparams)

        hparams.load_checkpoint_epoch = None
        hparams.epochs = 2
        hparams.batch_first = True
        hparams.batch_size_train = 2
        hparams.batch_size_val = hparams.batch_size_train
        hparams.epochs_per_checkpoint = 2
        hparams.model_type = "RNNDYN-1_RELU_32-1_FC_67"
        model_config = NamedForwardWrapper.Config(
            wrapped_model_config=rnn_dyn.convert_legacy_to_config(in_dim=hparams.num_questions, hparams=hparams),
            input_names=['questions'],
            output_names=['pred_acoustic_features'],
            batch_first=hparams.batch_first)
        hparams.optimiser_args["lr"] = 0.01
        hparams.scheduler_type = "Plateau"
        hparams.use_best_as_final_model = False

        trainer.init(hparams=hparams, model_config=model_config, loss_configs=loss_configs, data_reader_configs=self._get_data_reader_configs(hparams))
        # trainer.save_checkpoint(hparams, model_path=os.path.join("integration", "fixtures", "test_model_in409_out67", hparams.networks_dir))  # Save fixture for e0.

        all_loss, all_loss_train, _ = trainer.train(hparams)
        self.assertEqual(len(loss_configs), len(all_loss))
        self.assertEqual(len(loss_configs), len(all_loss_train))
        expected_value = 1.64315
        actual_value = sum([losses[-1] for losses in all_loss.values()])
        self.assertTrue(numpy.isclose(expected_value, actual_value),
                        msg="Expected combined loss in last epoch to be {} but got {}."
                            .format(expected_value, actual_value))

        log_dir = os.path.join(hparams.out_dir, hparams.model_name, "tensorboard")
        self.assertEqual(4, len([name for name in os.listdir(log_dir)
                                 if os.path.isdir(os.path.join(log_dir, name))]))

        # trainer.save_checkpoint(hparams, model_path=os.path.join("integration", "fixtures", "test_model_in409_out67_tmp", hparams.networks_dir))  # Save with optimiser.
        shutil.rmtree(hparams.out_dir, ignore_errors=True)

    def test_train_batch_first(self):

        hparams = self._get_hparams()
        hparams.seed = 42
        hparams.batch_first = True
        hparams.out_dir = os.path.join(hparams.out_dir, "test_train_batch_first")  # Add function name to path.
        trainer = self._get_trainer(hparams)

        hparams.load_checkpoint_epoch = None
        hparams.epochs = 2
        hparams.batch_size_train = 2
        hparams.batch_size_test = 2
        hparams.batch_size_val = hparams.batch_size_train
        hparams.epochs_per_checkpoint = 2
        hparams.optimiser_args["lr"] = 0.01
        hparams.scheduler_type = "Plateau"
        hparams.use_best_as_final_model = False
        hparams.model_type = "RNNDYN-1_RELU_32-1_Conv1d_16_3-1_FC_67"

        loss_configs = self._get_loss_configs()
        model_config = NamedForwardWrapper.Config(
            wrapped_model_config=rnn_dyn.convert_legacy_to_config(in_dim=hparams.num_questions, hparams=hparams),
            input_names=['questions'],
            output_names=['pred_acoustic_features'],
            batch_first=hparams.batch_first)
        trainer.init(hparams=hparams, model_config=model_config, loss_configs=loss_configs, data_reader_configs=self._get_data_reader_configs(hparams))

        all_loss, all_loss_train, _ = trainer.train(hparams)
        # self.assertTrue(numpy.isclose(0.91993, all_loss[-1]))
        expected_value = 0.926471
        actual_value = sum([losses[-1] for losses in all_loss.values()])
        self.assertTrue(numpy.isclose(expected_value, actual_value),
                        msg="Expected combined loss in last epoch to be {} but got {}."
                            .format(expected_value, actual_value))

        log_dir = os.path.join(hparams.out_dir, hparams.model_name, "tensorboard")
        self.assertEqual(2, len([name for name in os.listdir(log_dir)
                                 if os.path.isdir(os.path.join(log_dir, name))]))

        shutil.rmtree(hparams.out_dir, ignore_errors=True)

    def test_train_e4_plus2x2(self):
        logging.basicConfig(level=logging.INFO)

        for seed in [109]:  # itertools.count(0):
            hparams = self._get_hparams()
            hparams.out_dir = os.path.join(hparams.out_dir, "test_train_e4_plus2")  # Add function name to path.
            hparams.seed = seed
            trainer = self._get_trainer(hparams)

            hparams.model_type = "TestArchitecture"
            hparams.epochs = 4
            hparams.batch_size_train = 2
            hparams.batch_size_val = hparams.batch_size_train
            hparams.epochs_per_checkpoint = 2
            hparams.optimiser_args["lr"] = 0.01
            hparams.scheduler_type = "Plateau"
            hparams.use_best_as_final_model = False
            hparams.model_type = "RNNDYN-1_RELU_32-1_FC_67"

            loss_configs = self._get_loss_configs()
            model_config = NamedForwardWrapper.Config(
                wrapped_model_config=rnn_dyn.convert_legacy_to_config(in_dim=hparams.num_questions, hparams=hparams),
                input_names=['questions'],
                output_names=['pred_acoustic_features'],
                batch_first=hparams.batch_first)
            trainer.init(hparams=hparams, model_config=model_config, loss_configs=loss_configs,
                         data_reader_configs=self._get_data_reader_configs(hparams))

            all_loss, all_loss_train, _ = trainer.train(hparams)

            all_loss = list(all_loss.values())[0]
            all_loss_train = list(all_loss_train.values())[0]
            # if all_loss[-1] > min(all_loss) and all_loss_train[-1] < all_loss_train[1 if hparams.start_with_test else 0]:
            #     break

        # print(all_loss)
        # print(seed)

        # Training loss decreases?
        self.assertLess(all_loss_train[-1],
                        all_loss_train[1 if hparams.start_with_test else 0],
                        msg="Training loss did not decrease over {} epochs: {}".format(hparams.epochs, all_loss_train))

        # Expected number of checkpoints saved?
        checkpoint_dir = os.path.join(hparams.out_dir, hparams.model_name, hparams.networks_dir)
        self.assertEqual(int(hparams.epochs / hparams.epochs_per_checkpoint) + 1,
                         len([name for name in os.listdir(checkpoint_dir)
                              if os.path.isfile(os.path.join(checkpoint_dir, name))
                              and "params_e" in name]))

        # Best model saved?
        best_model_path = os.path.join(hparams.out_dir, hparams.model_name, hparams.networks_dir, "params_best")
        self.assertTrue(os.path.isfile(best_model_path))

        # Final model is not best model.
        self.assertFalse(filecmp.cmp(best_model_path,
                                     os.path.join(checkpoint_dir, "params_e{}".format(hparams.epochs)), False),
                         msg="Best model should not be the same as model of last epoch.")

        # Try reloading and training.
        hparams.load_checkpoint_epoch = -1
        hparams.load_optimiser = False
        hparams.start_with_test = True
        hparams.use_best_as_final_model = True
        trainer = self._get_trainer(hparams)
        trainer.init(hparams=hparams, model_config=model_config, loss_configs=loss_configs, data_reader_configs=self._get_data_reader_configs(hparams))
        hparams.epochs = 2
        hparams.epochs_per_test = 2
        trainer.train(hparams)
        previous_weights = trainer.model_handler.model.layer_groups[0][0].weight.detach().cpu().numpy().copy()
        trainer.train(hparams)
        current_weights = trainer.model_handler.model.layer_groups[0][0].weight.detach().cpu().numpy()
        self.assertTrue((previous_weights != current_weights).any(), "Weights did not change during training.")

        shutil.rmtree(hparams.out_dir, ignore_errors=True)

    def test_train_e4_save_best_plus2(self):
        # logging.basicConfig(level=logging.INFO)
        hparams = self._get_hparams()
        hparams.out_dir = os.path.join(hparams.out_dir, "test_train_e4_save_best")  # Add function name to path.
        hparams.seed = 1

        trainer = self._get_trainer(hparams)

        hparams.model_type = "RNNDYN-1_RELU_32-1_FC_67"
        hparams.epochs = 4
        hparams.batch_size_train = 2
        hparams.batch_size_val = hparams.batch_size_train
        hparams.epochs_per_checkpoint = 2
        hparams.optimiser_args["lr"] = 0.001
        hparams.scheduler_type = "Plateau"
        hparams.use_best_as_final_model = True

        loss_configs = self._get_loss_configs()
        model_config = NamedForwardWrapper.Config(
            wrapped_model_config=rnn_dyn.convert_legacy_to_config(in_dim=hparams.num_questions, hparams=hparams),
            input_names=['questions'],
            output_names=['pred_acoustic_features'],
            batch_first=hparams.batch_first)
        # for seed in itertools.count():
            # hparams.seed = seed
        trainer.init(hparams=hparams, model_config=model_config, loss_configs=loss_configs, data_reader_configs=self._get_data_reader_configs(hparams))
        _, all_loss_train, _ = trainer.train(hparams)

        all_loss_train = list(all_loss_train.values())[0]
            # if min(all_loss_train) < all_loss_train[-1]:
            #     break
        assert min(all_loss_train) < all_loss_train[-1]

        # Final model is best model?
        checkpoint_dir = os.path.join(hparams.out_dir, hparams.model_name, hparams.networks_dir)
        self.assertTrue(equal_iterable(torch.load(os.path.join(checkpoint_dir, "params_best"))["params"],
                                       trainer.model_handler.model.state_dict()),
                        msg="Current model is not the same as best model but hparams.use_best_as_final_model is True.")
        # Final model is not last model?
        self.assertTrue(equal_iterable(
                            torch.load(os.path.join(checkpoint_dir, "params_e{}".format(hparams.epochs)))["params"],
                            trainer.model_handler.model.state_dict()),
                        msg="Saved model is the same as final checkpoint which should not be the best model.")

        hparams.epochs = 2
        previous_weights = trainer.model_handler.model.layer_groups[0][0].weight.detach().cpu().numpy().copy()
        trainer.train(hparams)
        current_weights = trainer.model_handler.model.layer_groups[0][0].weight.detach().cpu().numpy()
        self.assertTrue((previous_weights != current_weights).any(), "Weights did not change during training.")

        shutil.rmtree(hparams.out_dir)

    def test_train_e4_reload_best_loss(self):
        """Test early stopping when saved with load_last_checkpoint."""
        hparams = self._get_hparams()
        # Add function name to path.
        hparams.out_dir = os.path.join(hparams.out_dir,
                                       "test_train_e4_reload_best_loss")
        hparams.seed = 1
        hparams.epochs = 4
        hparams.batch_size_train = 2
        hparams.batch_size_val = hparams.batch_size_train
        hparams.epochs_per_checkpoint = 2
        hparams.optimiser_args["lr"] = 0.001
        hparams.scheduler_type = "Plateau"
        hparams.use_best_as_final_model = False
        hparams.load_newest_checkpoint = True

        loss_configs = self._get_loss_configs()
        data_reader_configs = self._get_data_reader_configs(hparams)
        model_config = NamedForwardWrapper.Config(
            input_names=data_reader_configs[1].name,
            batch_first=True,
            output_names=["pred_acoustic_features"],
            wrapped_model_config=rnn_dyn.Config(
                batch_first=True,
                in_dim=hparams.num_questions,
                hparams=hparams,
                layer_configs=[
                    rnn_dyn.Config.LayerConfig(
                        layer_type="Linear",
                        nonlin="ReLU",
                        out_dim=32,
                    ),
                    rnn_dyn.Config.LayerConfig(
                        layer_type="Linear",
                        out_dim=67
                    )
                ]
            )
        )

        # First training iteration.
        trainer = self._get_trainer(hparams)
        trainer.init(hparams=hparams, model_config=model_config,
                     loss_configs=loss_configs,
                     data_reader_configs=data_reader_configs)

        trainer.model_handler.test = Mock(
            side_effect=itertools.chain(
                [{loss_configs[0].name: numpy.array(1.0)}],
                itertools.repeat({loss_configs[0].name: numpy.array(10.0)})))

        trainer.train(hparams)
        first_iter_best_loss = trainer.best_loss

        # Second training iteration.
        trainer = self._get_trainer(hparams)
        trainer.init(hparams=hparams, model_config=model_config,
                     loss_configs=loss_configs,
                     data_reader_configs=data_reader_configs)

        trainer.model_handler.test = Mock(side_effect=itertools.repeat(
            {loss_configs[0].name: numpy.array(5.0)}))

        trainer.train(hparams)
        second_iter_best_loss = trainer.best_loss

        self.assertEqual(first_iter_best_loss, second_iter_best_loss,
                         msg="Best loss was not kept after reloading.")

        # Third training iteration. Don't load best_loss with optimiser.
        hparams.load_optimiser = False
        trainer = self._get_trainer(hparams)
        trainer.init(hparams=hparams, model_config=model_config,
                     loss_configs=loss_configs,
                     data_reader_configs=data_reader_configs)

        trainer.model_handler.test = Mock(side_effect=itertools.repeat(
            {loss_configs[0].name: numpy.array(8.0)}))

        trainer.train(hparams)
        third_iter_best_loss = trainer.best_loss

        self.assertEqual(numpy.array(8.0), third_iter_best_loss,
                         msg="Best loss was kept after reloading.")

        shutil.rmtree(hparams.out_dir)

    def test_train_no_best_model(self):
        # Load a model, and use a mock so that test loss goes up. Check if no best model is there.
        hparams = self._get_hparams()
        hparams.out_dir = os.path.join(hparams.out_dir, "test_train_no_best_model")  # Add function name to path
        hparams.model_name = "test_model_in409_out67"
        hparams.model_path = os.path.join("integration", "fixtures", hparams.model_name, hparams.networks_dir)
        hparams.load_from_checkpoint = True
        hparams.checkpoint_epoch = 0
        hparams.load_optimiser = False
        hparams.seed = 0
        hparams.batch_size_train = 2
        hparams.batch_size_val = hparams.batch_size_train
        hparams.epochs = 1
        hparams.use_best_as_final_model = True
        hparams.optimiser_args["lr"] = 0.001
        hparams.scheduler_type = "Plateau"
        hparams.start_with_test = True

        loss_configs = self._get_loss_configs()
        trainer = self._get_trainer(hparams)
        trainer.init(hparams=hparams, loss_configs=loss_configs, data_reader_configs=self._get_data_reader_configs(hparams))
        trainer.model_handler.test = Mock(
            side_effect=itertools.chain([{loss_configs[0].name: numpy.array(1.0)}],
                                        itertools.repeat({loss_configs[0].name: numpy.array(10.0)})))

        all_loss, *_ = trainer.train(hparams)
        with unittest.mock.patch.object(trainer.logger, "warning") as mock_logger:
            trainer.train(hparams)
            mock_logger.assert_called_with("No best model exists. Continue with current one.")

        shutil.rmtree(hparams.out_dir)

    def test_train_e2_with_fixed_layers(self):
        # logging.basicConfig(level=logging.INFO)
        hparams = self._get_hparams()
        hparams.out_dir = os.path.join(hparams.out_dir, "test_train_e2_with_fixed_layers")  # Add function name to path
        hparams.model_name = "test_model_in409_out67"
        hparams.model_path = os.path.join("integration", "fixtures", hparams.model_name, hparams.networks_dir)
        hparams.load_optimiser = False
        hparams.seed = 0
        hparams.batch_size_train = 2
        hparams.batch_size_val = hparams.batch_size_train
        hparams.epochs = 1
        hparams.load_checkpoint_epoch = 0
        hparams.use_best_as_final_model = True
        hparams.optimiser_args["lr"] = 0.001
        hparams.scheduler_type = "Plateau"
        hparams.start_with_test = True

        loss_configs = self._get_loss_configs()
        trainer = self._get_trainer(hparams)
        trainer.init(hparams=hparams, loss_configs=loss_configs, data_reader_configs=self._get_data_reader_configs(hparams))
        for param in trainer.model_handler.model.model[0][0].parameters():
            param.requires_grad = False

        self.assertFalse(trainer.model_handler.model.model[0][0].weight.requires_grad)
        trainer.model_handler.test = Mock(
            side_effect=itertools.chain([{loss_configs[0].name: numpy.array(10.0)}],
                                        itertools.repeat({loss_configs[0].name: numpy.array(1.0)})))

        trainer.train(hparams)
        self.assertFalse(trainer.model_handler.model.model[0][0].weight.requires_grad,
                         msg="requires_grad was changed during training.")

        shutil.rmtree(hparams.out_dir)

    def test_named_loss_seq_mask(self):
        # Load a model, and use a mock so that test loss goes up. Check if no best model is there.
        hparams = self._get_hparams()
        hparams.out_dir = os.path.join(hparams.out_dir, "test_named_loss_seq_mask")  # Add function name to path
        hparams.model_name = "test_model"
        hparams.batch_size_train = 2
        hparams.batch_size_val = hparams.batch_size_train
        hparams.epochs = 1
        hparams.seed = 0
        hparams.use_best_as_final_model = True
        hparams.optimiser_args["lr"] = 0.001
        hparams.scheduler_type = "Plateau"
        hparams.start_with_test = True
        hparams.n_frames_per_step = 1

        loss_configs = self._get_loss_configs()
        loss_configs[0].type = "L1Loss"
        loss_configs[0].reduction = "sum"
        hparams.val_set_perc = 0.3
        trainer = self._get_trainer(hparams)

        model_config = NamedForwardWrapper.Config(
            wrapped_model_config=rnn_dyn.Config(
                layer_configs=[],
                in_dim=67
            ),
            batch_first=hparams.batch_first,
            input_names=["acoustic_features"],
            output_names=["pred_acoustic_features"]
        )
        trainer.init(hparams=hparams, loss_configs=loss_configs, model_config=model_config, data_reader_configs=self._get_data_reader_configs(hparams))

        shutil.rmtree(hparams.out_dir)

    def test_vaekld_loss(self):
        # Load a model, and use a mock so that test loss goes up. Check if no best model is there.
        hparams = self._get_hparams()
        hparams.out_dir = os.path.join(hparams.out_dir, "test_vaekld_loss")  # Add function name to path
        hparams.model_name = "test_model"
        hparams.batch_size_train = 2
        hparams.batch_size_val = hparams.batch_size_train
        hparams.epochs = 1
        hparams.seed = 0
        hparams.use_best_as_final_model = True
        hparams.optimiser_args["lr"] = 0.001
        hparams.scheduler_type = "Plateau"
        hparams.start_with_test = True
        hparams.n_frames_per_step = 1

        loss_configs = [
            VAEKLDLoss.Config(
                name="VAEKLD_loss",
                type_="VAEKLDLoss",
                input_names=["emb_mu", "emb_logvar"],
                seq_mask="acoustic_features_mask",
                start_step=10,
                annealing_points=(-1, 100),
                annealing_steps=10
            )
        ]
        hparams.val_set_perc = 0.3
        trainer = self._get_trainer(hparams)

        model_config = NamedForwardWrapper.Config(
            wrapped_model_config=rnn_dyn.Config(
                layer_configs=[
                    rnn_dyn.Config.LayerConfig(
                        layer_type="VAE",
                        out_dim=4
                    )
                ],
                in_dim=67
            ),
            batch_first=hparams.batch_first,
            input_names=["acoustic_features"],
            output_names=["emb_z", "emb_mu", "emb_logvar"]
        )
        trainer.init(hparams=hparams, loss_configs=loss_configs, model_config=model_config, data_reader_configs=self._get_data_reader_configs(hparams))

        loss = trainer.test(hparams)["VAEKLD_loss"]
        self.assertEquals(loss, 0.0)

        trainer.total_steps = 10
        loss = trainer.test(hparams)["VAEKLD_loss"]
        self.assertGreater(loss, 0.0)

        shutil.rmtree(hparams.out_dir)

    def test_train_e3_ema(self):
        # logging.basicConfig(level=logging.INFO)

        hparams = self._get_hparams()
        hparams.out_dir = os.path.join(hparams.out_dir, "test_train_e3_ema")  # Add function name to path.
        hparams.seed = 1234
        trainer = self._get_trainer(hparams)

        hparams.model_type = "RNNDYN-1_RELU_32-1_FC_67"
        hparams.epochs = 1
        hparams.batch_size_train = len(self.id_list)
        hparams.batch_size_val = hparams.batch_size_train
        hparams.epochs_per_checkpoint = 1
        hparams.optimiser_args["lr"] = 0.01
        hparams.scheduler_type = "None"
        hparams.use_best_as_final_model = False
        hparams.start_with_test = True
        hparams.ema_decay = 0.9

        loss_configs = self._get_loss_configs()
        model_config = NamedForwardWrapper.Config(
            wrapped_model_config=rnn_dyn.convert_legacy_to_config(
                in_dim=hparams.num_questions, hparams=hparams),
            input_names=['questions'],
            output_names=['pred_acoustic_features'],
            batch_first=hparams.batch_first)
        trainer.init(hparams=hparams,
                     model_config=model_config,
                     loss_configs=loss_configs,
                     data_reader_configs=self._get_data_reader_configs(hparams))

        models = [copy.deepcopy(trainer.model_handler.model)]
        # print("Test model {}: {}".format(0, list(models[-1].parameters())[0]))
        all_loss_train = list()
        iterations = 4
        for epoch in range(iterations):
            tmp_all_loss, tmp_all_loss_train, _ = trainer.train(hparams)
            tmp_all_loss_train = list(tmp_all_loss_train.values())[0]
            all_loss_train.append(tmp_all_loss_train[-1])
            models.append(copy.deepcopy(trainer.model_handler.model))
            # print("Test model {}: {}".format(epoch + 1, list(models[-1].parameters())[0]))
            # print("EMA {}: {}".format(epoch + 1, list(trainer.model_handler.ema.model.parameters())[0]))

        # Expected number of checkpoints saved?
        checkpoint_dir = os.path.join(hparams.out_dir, hparams.model_name, hparams.networks_dir)
        self.assertEqual((int(hparams.epochs * iterations / hparams.epochs_per_checkpoint) + 2),  #*2 for ema model.
                         len([name for name in os.listdir(checkpoint_dir) if
                              os.path.isfile(os.path.join(checkpoint_dir, name)) and "params_" in name]))

        # EMA model is average of others.
        expected_weights = list(models[0].parameters())[0].detach()
        # print("Exp {}: {}".format(0, expected_weights))
        for index, model in enumerate(models[1:]):
            expected_weights = expected_weights * hparams.ema_decay + (1 - hparams.ema_decay) * list(model.parameters())[0].detach()
            # print("Exp {}: {}".format(index, expected_weights))
        self.assertTrue((list(trainer.model_handler.ema.model.parameters())[0] - expected_weights).abs().max() < 1E-5)

        # Try continue training.
        hparams.load_checkpoint_epoch = -1
        trainer = self._get_trainer(hparams)
        trainer.init(hparams,
                     loss_configs=self._get_loss_configs(),
                     data_reader_configs=self._get_data_reader_configs(hparams))
        trainer.train(hparams)

        shutil.rmtree(hparams.out_dir)

    def test_train_exponential_decay(self):
        # logging.basicConfig(level=logging.INFO)

        hparams = self._get_hparams()
        hparams.out_dir = os.path.join(hparams.out_dir, "test_train_exponential_decay")  # Add function name to path.
        hparams.epochs = 1
        # hparams.model_type = None
        hparams.load_checkpoint_step = 8
        target_dir = os.path.join(hparams.out_dir, hparams.model_name, hparams.networks_dir)
        makedirs_safe(target_dir)
        copy_tree(os.path.join("integration", "fixtures", "test_model_in409_out67", hparams.networks_dir), target_dir)
        # hparams.model_type = "RNNDYN-1_RELU_32-1_FC_{}".format(3 * hparams.num_coded_sps + 7)
        hparams.seed = 1234
        hparams.optimiser_args["lr"] = 0.1
        hparams.scheduler_type = "Exponential"
        hparams.scheduler_args["gamma"] = 0.9
        hparams.use_saved_learning_rate = True
        hparams.use_best_as_final_model = False

        trainer = self._get_trainer(hparams)
        loss_configs = self._get_loss_configs()
        trainer.init(hparams=hparams, loss_configs=loss_configs, data_reader_configs=self._get_data_reader_configs(hparams))
        # for group in trainer.model_handler.optimiser.param_groups:
        #     group.setdefault('initial_lr', hparams.optimiser_args["lr"])  # Add missing initial_lr to all groups.
        trainer.train(hparams)

        expected_lr = 0.001 * hparams.scheduler_args["gamma"]**(1 + hparams.epochs * len(trainer.id_list_train))
        self.assertTrue(numpy.isclose(expected_lr, trainer.model_handler.optimiser.param_groups[0]["lr"]),
                        "Exponential decay was not computed based on total number of epochs. "
                        "Which should be the case when hparams.use_saved_learning_rate=True.")

        # Try again with reset learning rate.
        trainer = self._get_trainer(hparams)
        hparams.use_saved_learning_rate = False
        trainer.init(hparams=hparams, loss_configs=loss_configs, data_reader_configs=self._get_data_reader_configs(hparams))
        # for group in trainer.model_handler.optimiser.param_groups:
        #     group.setdefault('initial_lr', hparams.optimiser_args["lr"])  # Add missing initial_lr to all groups.
        trainer.train(hparams)

        expected_lr = hparams.optimiser_args["lr"] * hparams.scheduler_args["gamma"] ** (len(trainer.id_list_train))
        self.assertTrue(numpy.isclose(expected_lr, trainer.model_handler.optimiser.param_groups[0]["lr"]),
                        "Exponential decay was not reset for this training loop, "
                        "which should be the case when hparam.use_saved_learning_rate=False.")

        shutil.rmtree(hparams.out_dir)

    def test_synth_wav(self):
        num_test_files = 2

        hparams = self._get_hparams()
        # Add function name to path.
        hparams.out_dir = os.path.join(hparams.out_dir, "test_synth_wav")
        hparams.model_name = "test_model_in409_out67"
        hparams.model_path = os.path.join("integration", "fixtures",
                                          hparams.model_name,
                                          hparams.networks_dir)
        hparams.load_checkpoint_epoch = 2
        hparams.synth_fs = 16000
        hparams.frame_size_ms = 5
        hparams.synth_ext = "wav"
        hparams.do_post_filtering = True
        hparams.batch_size_synth = 4

        trainer = self._get_trainer(hparams)
        trainer.init(
            hparams=hparams,
            data_reader_configs=self._get_data_reader_configs(hparams))
        # hparams.synth_dir = hparams.out_dir
        trainer.synth(hparams, self.id_list[:num_test_files],
                      post_processing_mapping=self.post_processing_mapping)
        expected_synth_dir = os.path.join(
            hparams.out_dir, hparams.model_name,
            "synth", "e" + str(hparams.load_checkpoint_epoch))
        found_files = list([
            name for name in os.listdir(expected_synth_dir)
            if os.path.isfile(os.path.join(expected_synth_dir, name))
            and name.endswith("_WORLD." + hparams.synth_ext)])

        # Check number of created files.
        self.assertEqual(len(self.id_list[:num_test_files]), len(found_files),
                         msg="Number of {} files in synth_dir directory does"
                         " not match.".format(hparams.synth_ext))

        # Check readability and length of one created file.
        raw, fs = soundfile.read(os.path.join(expected_synth_dir,
                                              found_files[0]))
        self.assertEqual(hparams.synth_fs, fs, msg="Desired sampling frequency"
                         " of output doesn't match.")
        first_id_name = [id_name for id_name in self.id_list[:num_test_files]
                         if id_name in found_files[0]][0]
        labels = trainer.get_labels(reader_name="cmp_features",
                                    id_name=first_id_name)
        expected_length = (len(raw) / hparams.synth_fs
                           / hparams.frame_size_ms * 1000)
        self.assertTrue(abs(expected_length - len(labels)) < 10,
                        msg="Saved raw audio file length does not roughly "
                        "match length of labels.")

        shutil.rmtree(hparams.out_dir)

    def test_synth_mp3(self):
        num_test_files = 2

        hparams = self._get_hparams()
        hparams.out_dir = os.path.join(hparams.out_dir, "test_synth_mp3")  # Add function name to path
        hparams.model_name = "test_model_in409_out67"
        hparams.model_path = os.path.join("integration", "fixtures", hparams.model_name, hparams.networks_dir)
        hparams.load_checkpoint_epoch = -1
        hparams.synth_fs = 16000
        hparams.frame_size_ms = 5
        hparams.synth_ext = "mp3"

        trainer = self._get_trainer(hparams)
        trainer.init(hparams, data_reader_configs=self._get_data_reader_configs(hparams))
        hparams.synth_dir = hparams.out_dir
        trainer.synth(hparams, self.id_list[:num_test_files], post_processing_mapping=self.post_processing_mapping)

        found_files = list([name for name in os.listdir(hparams.synth_dir)
                            if os.path.isfile(os.path.join(hparams.synth_dir, name))
                            and name.endswith("_WORLD." + hparams.synth_ext)])
        found_files_wav = list([name for name in os.listdir(hparams.synth_dir)
                                if os.path.isfile(os.path.join(hparams.synth_dir, name))
                                and name.endswith("_WORLD.wav")])
        # Check number of created files.
        self.assertEqual(len(self.id_list[:num_test_files]), len(found_files),
                         msg="Number of {} files in synth_dir directory does not match.".format(hparams.synth_ext))
        self.assertEqual(0, len(found_files_wav), msg="Did not expect any wav files in the synth_dir directory.")

        # Check readability and length of one created file, requires some steps because it is mp3.
        sound = pydub.AudioSegment.from_file(os.path.join(hparams.synth_dir, found_files[0]))
        bit_depth = sound.split_to_mono()[0].sample_width * 8
        array_type = pydub.utils.get_array_type(bit_depth)
        fs = bit_depth * 1000
        raw = array.array(array_type, sound.split_to_mono()[0]._data)

        self.assertEqual(hparams.synth_fs, fs, msg="Desired sampling frequency of output doesn't match.")
        first_id_name = [id_name for id_name in self.id_list[:num_test_files] if id_name in found_files[0]][0]
        labels = trainer.get_labels(reader_name="cmp_features", id_name=first_id_name)
        expected_length = len(raw) / hparams.synth_fs / hparams.frame_size_ms * 1000
        self.assertTrue(abs(expected_length - len(labels)) < 10,
                        msg="Saved raw audio file length does not roughly match length of labels.")

        shutil.rmtree(hparams.out_dir)

    def test_gen_figure(self):
        logging.basicConfig(level=logging.INFO)
        hparams = self._get_hparams()
        hparams.out_dir = os.path.join(hparams.out_dir, "test_gen_figure")
        hparams.model_name = "test_model_in409_out67"
        hparams.model_path = os.path.join(
            "integration", "fixtures", hparams.model_name, hparams.networks_dir)
        hparams.load_checkpoint_epoch = -1
        hparams.batch_size_gen_figure = 4

        trainer = self._get_trainer(hparams)
        trainer.init(hparams=hparams,
                     data_reader_configs=self._get_data_reader_configs(hparams))

        trainer.gen_figure_from_output = Mock(side_effect=itertools.repeat(None))
        trainer.gen_figure(hparams=hparams, ids_input=self.id_list,
                           post_processing_mapping=self.post_processing_mapping,
                           plotter_configs=None)
        self.assertEqual(trainer.gen_figure_from_output.call_count,
                         len(self.id_list))

        shutil.rmtree(hparams.out_dir)

    def test_copy_synth(self):
        num_test_files = 2

        hparams = self._get_hparams()
        hparams.out_dir = os.path.join(hparams.out_dir, "test_copy_synth")  # Add function name to path
        hparams.model_name = "test_model_in409_out67"
        hparams.model_path = os.path.join("integration", "fixtures", hparams.model_name, hparams.networks_dir)
        hparams.load_checkpoint_epoch = -1
        hparams.synth_fs = 16000
        hparams.frame_size_ms = 5
        hparams.synth_ext = "mp3"
        hparams.add_hparam("synth_feature_names", "acoustic_features")
        hparams.add_hparam("add_deltas", True)

        trainer = self._get_trainer(hparams)
        trainer.init(hparams, data_reader_configs=self._get_data_reader_configs(hparams))
        hparams.synth_dir = hparams.out_dir
        trainer.copy_synth(hparams=hparams, id_list=self.id_list[:num_test_files])

        found_files = list([name for name in os.listdir(hparams.synth_dir)
                            if os.path.isfile(os.path.join(hparams.synth_dir, name))
                            and name.endswith("_ref_{}{}_WORLD.{}".format(hparams.num_coded_sps, hparams.sp_type, hparams.synth_ext))
                            and hparams.model_name not in name])
        found_files_wav = list([name for name in os.listdir(hparams.synth_dir)
                                if os.path.isfile(os.path.join(hparams.synth_dir, name))
                                and name.endswith("_ref_{}{}_WORLD.wav".format(hparams.num_coded_sps, hparams.sp_type))])
        # Check number of created files.
        self.assertEqual(len(self.id_list[:num_test_files]), len(found_files),
                         msg="Number of {} files in synth_dir directory does not match.".format(hparams.synth_ext))
        self.assertEqual(0, len(found_files_wav), msg="Did not expect any wav files in the synth_dir directory.")

        # Check readability and length of one created file, requires some steps because it is mp3.
        sound = pydub.AudioSegment.from_file(os.path.join(hparams.synth_dir, found_files[0]))
        bit_depth = sound.split_to_mono()[0].sample_width * 8
        array_type = pydub.utils.get_array_type(bit_depth)
        fs = bit_depth * 1000
        raw = array.array(array_type, sound.split_to_mono()[0]._data)

        self.assertEqual(hparams.synth_fs, fs, msg="Desired sampling frequency of output doesn't match.")
        first_id_name = [id_name for id_name in self.id_list[:num_test_files] if id_name in found_files[0]][0]
        labels = trainer.get_labels(reader_name="cmp_features", id_name=first_id_name)
        expected_length = len(raw) / hparams.synth_fs / hparams.frame_size_ms * 1000
        self.assertTrue(abs(expected_length - len(labels)) < 10,
                        msg="Saved raw audio file length does not roughly match length of labels.")

        shutil.rmtree(hparams.out_dir)


# TODO: Test with GPU
# TODO: Multispeaker tests.
# TODO: Everything with batch_first=True
