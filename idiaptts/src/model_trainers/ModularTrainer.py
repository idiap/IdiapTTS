#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#

from contextlib import ExitStack
import copy
from datetime import timedelta, datetime
from functools import partial
import logging
import math
import platform
from operator import itemgetter
import os
import random
import resource
from shutil import copy2
import sys
from timeit import default_timer as timer
from typing import Union, Any, List, Optional, cast, Dict, Tuple

import numpy as np
from torchinfo import summary

from idiaptts.misc.utils import log_git_hash, makedirs_safe, get_gpu_memory_map
from idiaptts.src.DataPlotter import DataPlotter
import idiaptts.src.data_preparation as datasets
from idiaptts.src.data_preparation.PyTorchDatareadersDataset import PyTorchDatareadersDataset
from idiaptts.src.data_preparation.NpzDataReader import DataReader
from idiaptts.src.ExtendedHParams import ExtendedHParams
from idiaptts.src.neural_networks.pytorch.loss.NamedLoss import NamedLoss
from idiaptts.src.neural_networks.pytorch.ModularModelHandlerPyTorch import ModularModelHandlerPyTorch as ModelHandler
# from idiaptts.src.neural_networks.pytorch.models.EncDecDyn import *
from idiaptts.src.Synthesiser import Synthesiser


class ModularTrainer(object):

    logger = logging.getLogger(__name__)

    def __init__(self,
                 hparams: ExtendedHParams,
                 id_list: List[str],
                 data_reader_configs: List[DataReader.Config] = None):

        self.logger.info("Running on host {}.".format(platform.node()))
        log_git_hash()
        self.tb_writer = None  # TensorBoard summary writer variable.

        assert (hparams is not None)
        hparams.enable_backwards_compatibility()
        self._check_gpus(hparams)

        if hparams.seed is not None:
            ModelHandler.seed(hparams.seed)  # Seed the backend.
            np.random.seed(hparams.seed)
            random.seed(hparams.seed)

        self._setup_id_lists(id_list, hparams)

        self.logger.info("Create ModularHandler.")
        self.model_handler = ModelHandler()

        self.batch_collate_fn = None
        self.batch_decollate_fn = self.split_batch
        self.train_losses = []
        self.validation_losses = []
        self.reset_best_loss()

        # Legacy support
        if data_reader_configs is not None:
            self.logger.warn("data_reader_configs should be given as parameter "
                             "of init(...) call.", DeprecationWarning)
            self._data_reader_configs = data_reader_configs
        else:
            self._data_reader_configs = None

    def _check_gpus(self, hparams: ExtendedHParams):
        if hparams.use_gpu:
            if hparams.num_gpus > 1:
                os.environ['CUDA_VISIBLE_DEVICES'] = str(tuple(range(hparams.num_gpus)))
                assert (hparams.batch_size % hparams.num_gpus == 0), "Batch of size {} is not equally dividable into the given number of GPUs ({}).".format(hparams.batch_size, hparams.num_gpus)
            if ModelHandler.cuda_is_available():
                device_count = ModelHandler.device_count()
                if not device_count == hparams.num_gpus:
                    self.logger.error("Specified GPU count in hparams.num_gpus ({}) doesn't match hardware ({})."
                                      .format(hparams.num_gpus, device_count))
                assert (device_count == hparams.num_gpus), "Specified GPU count doesn't match hardware."
            else:
                self.logger.warning("No CUDA device available, use CPU mode instead.")
                hparams.use_gpu = False

    def _setup_id_lists(self, id_list: List[str], hparams: ExtendedHParams):
        if not hasattr(self, "id_list_train") or self.id_list_train is None:
            id_list_shuffled = id_list
            if hparams.seed is not None:
                id_list_shuffled = random.sample(id_list, len(id_list))

            # Partition (randomly sorted) ids into [val_set, train_set, test_set].
            assert (hparams.test_set_perc + hparams.val_set_perc < 1)
            if hparams.val_set_perc > 0.0:
                num_valset = max(1, int(len(id_list_shuffled) * hparams.val_set_perc))
                self.id_list_val = id_list_shuffled[:num_valset]
            else:
                num_valset = 0
                self.id_list_val = None
            if hparams.test_set_perc > 0.0:
                num_testset = max(1, int(len(id_list_shuffled) * hparams.test_set_perc))
                self.id_list_test = id_list_shuffled[-num_testset:]
            else:
                num_testset = 0
                self.id_list_test = None
            self.id_list_train = id_list_shuffled[num_valset:-num_testset] if num_testset > 0 \
                else id_list_shuffled[num_valset:]
            assert (len(self.id_list_train) > 0)

    def reset_best_loss(self):
        self.best_loss = np.nan

    @staticmethod
    def create_hparams(hparams_string=None, verbose=False):
        """Create model hyper-parameters. Parse non-default from given string."""
        return ExtendedHParams.create_hparams(hparams_string, verbose)

    @staticmethod
    def split_batch(data: Dict[str, np.ndarray], seq_lengths: Dict[str, int], batch_first=True):
        return {k: ModularTrainer._split_return_values(v, seq_lengths[k], batch_first=batch_first)
                for k, v in data.items()}

    @classmethod
    def _split_return_values(cls, input_values, seq_length_output, permutation=None, batch_first=False):
        if input_values is None:
            return None

        # Special case for bidirectional layers where the hidden state is a tuple.
        if isinstance(input_values, tuple):
            # If hidden is a tuple of None, return it directly.
            if all(v is None for v in input_values):
                return input_values

            # Split hidden states in their batch dimension.
            tuple_splitted = tuple(
                map(lambda x: cls._split_return_values(x, seq_length_output, permutation, batch_first),
                    input_values))

            # Now sort into each batch.
            return_values = list()
            batch_size = len([t for t in tuple_splitted if t is not None][0])  # Get batch size from not None element.

            for index in range(batch_size):
                batch = list()
                for element in tuple_splitted:
                    if element is None or (isinstance(element, tuple) and all(v is None for v in element)):
                        batch.append(element)  # Handles None and tuples of None.
                    else:
                        batch.append(element[index])
                return_values.append(tuple(batch))

            return tuple(return_values)

        if not isinstance(input_values, np.ndarray):
            cls.logger.error("Expected numpy tensor but input is of type {}.".format(type(input_values)))
            raise TypeError()

        # Return value is tensor.
        if batch_first:
            return_values = np.split(input_values, input_values.shape[0], axis=0)
            return_values = list(map(partial(np.squeeze, axis=0), return_values))
        else:
            return_values = np.split(input_values, input_values.shape[1], axis=1)
            return_values = list(map(partial(np.squeeze, axis=1), return_values))

        if seq_length_output is not None and len(seq_length_output) > 1:
            for idx in range(len(return_values)):
                return_values[idx] = return_values[idx][:seq_length_output[idx]]

        if permutation is not None:
            return_values_unsorted = return_values.copy()
            for org_index, current_index in enumerate(permutation):
                return_values_unsorted[current_index] = return_values[org_index]
            return_values = return_values_unsorted

        return return_values

    def init(self,
             hparams: ExtendedHParams,
             model_config=None,
             loss_configs: List[NamedLoss.Config] = None,
             data_reader_configs: List[DataReader.Config] = None):

        self.log_memory(hparams.use_gpu)
        assert hparams.has_value("model_name"), "hparams.model_name is required."
        makedirs_safe(os.path.join(hparams.out_dir, hparams.model_name,
                                   hparams.networks_dir))

        try:
            from torch.utils.tensorboard import SummaryWriter

            if hparams.has_value("tensorboard_dir"):
                tensorboard_dir = hparams.tensorboard_dir
            else:
                tensorboard_dir = os.path.join(hparams.out_dir,
                                               hparams.model_name,
                                               "tensorboard")
            self.tb_writer = SummaryWriter(log_dir=tensorboard_dir)
            self.logger.info(
                "Started TensorBoard logging. Start the board with 'tensorboard"
                " --logdir {} --window_title {}' and use http://localhost:"
                "<automatic_port> to access.".format(
                    os.path.realpath(tensorboard_dir), hparams.model_name))
        except ImportError:
            self.tb_writer = None

        self.datareaders = dict()
        if data_reader_configs is not None:
            self._data_reader_configs = copy.deepcopy(data_reader_configs)
        elif self._data_reader_configs is not None:
            # External changes don't have an effect anymore.
            self._data_reader_configs = copy.deepcopy(self._data_reader_configs)
        else:
            raise ValueError("Parameter data_reader_configs is required.")
        self._setup_datareaders(self._data_reader_configs, hparams)

        if hparams.load_newest_checkpoint:
            try:
                self.load_checkpoint(hparams)
            except FileNotFoundError:
                self.logger.warning("No newest checkpoint found, creating a "
                                    "new model instead.")
                self.create_model(model_config, use_gpu=hparams.use_gpu)
                if hparams.epochs > 0:
                    self.save_checkpoint(hparams)
        elif (hparams.has_value("load_checkpoint_epoch")
                or hparams.has_value("load_checkpoint_step")):
            self.load_checkpoint(hparams)
        else:
            if model_config is not None:
                self.create_model(model_config, use_gpu=hparams.use_gpu)
            else:
                assert self.model_handler is not None, \
                    "Model config is required or model must already be created"
                assert self.model_handler.model is not None, \
                    "Model config is required or model must already be created"
            if hparams.epochs > 0:
                self.save_checkpoint(hparams)

        self._setup_loss_modules(loss_configs, use_gpu=hparams.use_gpu)

        self.logger.info("ModularTrainer ready.")

    def log_memory(self, use_gpu):
        self.logger.info("CPU memory: {} MB.".format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e3))
        if use_gpu:
            self.logger.info("GPU memory: {} MB.".format(get_gpu_memory_map()))

    def _setup_datareaders(self, data_reader_configs: List[DataReader.Config], hparams: ExtendedHParams):
        readers = list()
        for config in data_reader_configs:
            reader = config.create_reader()
            readers.append(reader)
            for out_name in reader.output_names:
                self.datareaders[out_name] = reader

        self.dataset_train = self.get_dataset(self.id_list_train, readers,
                                                 hparams, is_train_set=True)
        if self.id_list_val is not None:
            val_train_set_overlap = [id_name for id_name in self.id_list_val if id_name in self.id_list_train]
            assert len(val_train_set_overlap) == 0, "Found same ids in train and validation set: " \
                                                    + ", ".join(val_train_set_overlap)
            self.dataset_val = self.get_dataset(self.id_list_val, readers,
                                                   hparams, is_val_set=True)
        if self.id_list_test is not None:
            test_train_set_overlap = [id_name for id_name in self.id_list_test if id_name in self.id_list_train]
            assert len(test_train_set_overlap) == 0, "Found same ids in train and test set: " \
                                                     + ", ".join(test_train_set_overlap)
            if self.id_list_val is not None:
                test_val_set_overlap = [id_name for id_name in self.id_list_test if id_name in self.id_list_val]
                if len(test_val_set_overlap) > 0:
                    logging.error("Found same ids in val and test set: "
                                  + ", ".join(test_val_set_overlap))
                # assert len(test_val_set_overlap) == 0, "Found same ids in val and test set: " \
                #                                        + ", ".join(test_val_set_overlap)
            self.dataset_test = self.get_dataset(self.id_list_test, readers,
                                                    hparams, is_test_set=True)

    def get_dataset(self, id_list: List[str], datareaders: List,
                    hparams: ExtendedHParams, is_train_set: bool = False,
                    is_val_set: bool = False, is_test_set: bool = False):
        # self.dataset_train = PyTorchDatareaderDataset(self.id_list_train, readers, hparams)
        # self.dataset_train = PyTorchWindowingDatareadersDataset(self.id_list_train, readers, hparams, hparams.batch_size_train)
        dataset_class = getattr(datasets, hparams.dataset_type)
        return dataset_class(id_list, datareaders, hparams,
                             is_train_set=is_train_set,
                             is_val_set=is_val_set,
                             is_test_set=is_test_set)

    def create_model(self, model_config, use_gpu):
        assert model_config is not None, "Model config is required."
        self.model_handler.create_model(model_config, use_gpu=use_gpu)
        self.total_epoch = 0
        self.total_steps = 0

    def save_checkpoint(self,
                        hparams: ExtendedHParams,
                        model_path: os.PathLike = None,
                        save_as_best_model: bool = False,
                        save_as_last_model: bool = False):
        if model_path is None:
            model_path = self.get_model_path(hparams, ignore_model_path=True)
        self.model_handler.save_checkpoint(
            model_path=model_path,
            best_loss=self.best_loss,
            epoch=self.total_epoch,
            step=self.total_steps,
            save_as_best_model=save_as_best_model,
            save_as_epoch=hparams.epochs_per_checkpoint > 0,
            save_as_last_model=save_as_last_model,
            save_as_step=hparams.steps_per_checkpoint > 0)

    @staticmethod
    def get_model_path(hparams, ignore_model_path: bool = False):
        if hparams.model_path is None or ignore_model_path:
            assert hparams.out_dir is not None
            assert hparams.networks_dir is not None
            assert hparams.model_name is not None, "A model_name has to be given. No default exists."
            return os.path.join(hparams.out_dir, hparams.model_name, hparams.networks_dir)
        else:
            return hparams.model_path

    def load_best_model(self, hparams, model_path=None):
        if model_path is None:
            model_path = self.get_model_path(hparams, ignore_model_path=True)
        try:
            loss_epoch_step = self.model_handler.load_checkpoint(
                hparams=hparams,
                model_path=model_path,
                ignore_layers=False,  # Load all layers of the best model.
                load_optimiser=True,  # Load the optimiser state of the best model.
                load_scheduler=True,
                load_best_model=True)
            self.best_loss, self.total_epoch, self.total_steps = loss_epoch_step
            self.model_handler.ema = None  # Reset current exponential moving average model.
            self.logger.info("Using best (epoch {}) as final model.".format(
                self.total_epoch))
        except FileNotFoundError:
            self.logger.warning("No best model exists. Continue with current one.")

    def load_checkpoint(self, hparams, model_path=None):
        if model_path is None:
            model_path = self.get_model_path(hparams)
        try:
            loss_epoch_step = self.model_handler.load_checkpoint(
                hparams=hparams,
                model_path=model_path,
                epoch=hparams.load_checkpoint_epoch if hparams.has_value(
                    "load_checkpoint_epoch") else None,
                ignore_layers=True,
                load_optimiser=hparams.load_optimiser,
                load_scheduler=hparams.load_scheduler,
                step=hparams.load_checkpoint_step if hparams.has_value(
                    "load_checkpoint_step") else None)
            self.best_loss, self.total_epoch, self.total_steps = loss_epoch_step
        except FileNotFoundError as e:
            self.logger.error("Model does not exist at {}. {}".format(
                model_path, e))
            raise

    def _setup_loss_modules(self, loss_configs: List[NamedLoss.Config], use_gpu: bool = False):
        if loss_configs is None:
            return
        if type(loss_configs) not in [tuple, list]:
            loss_configs = [loss_configs]
        self.loss_modules = [config.create_loss() for config in loss_configs]
        if use_gpu:
            self.loss_modules = [loss.cuda() for loss in self.loss_modules]

    def train(self, hparams):
        """
        Train the model. Use generators for data preparation and model_handler for access.
        Generators have to be set in constructor of subclasses.

        :param hparams:          Hyper-parameter container.
        :return:                 A tuple of (all test loss, all training loss, the model_handler object).
        """
        self.sanity_check_train(hparams)
        self.logger.info(hparams.get_debug_string())
        network_summary = summary(self.model_handler.model, depth=100, verbose=0)
        self.logger.info(network_summary)
        if self.tb_writer is not None:
            self.tb_writer.add_text("HParams", "<pre>" + hparams.get_debug_string() + "</pre>")
            # self.tb_writer.add_graph(self.model_handler.model)  # This would require an input.
            self.tb_writer.add_text("Network", "<pre>" + str(network_summary) + "</pre>")

        # Skip training if epochs is not greater 0.
        if hparams.epochs <= 0:
            self.logger.info("Number of training epochs is {}. Skipping training.".format(hparams.epochs))
            return list(), list(), self.model_handler

        self.logger.info("Training set size: {}".format(len(self.id_list_train)))
        if self.id_list_val is not None and len(self.id_list_val) > 0:
            self.log_validation_set()
        self.log_test_set()

        # Setup components.
        if self.total_epoch is None:  # TODO: remove again
            self.total_epoch = self.total_steps // len(self.dataset_train)
        elif self.total_steps is None:
            self.total_steps = self.total_epoch * len(self.dataset_train)
        self.model_handler.set_dataset(hparams, self.dataset_train, self.dataset_val, self.batch_collate_fn)
        self.model_handler.set_optimiser(hparams)
        self.model_handler.set_scheduler(hparams,
                                         self.total_epoch if hparams.use_saved_learning_rate else 0,
                                         self.total_steps if hparams.use_saved_learning_rate else 0)
        self.model_handler.set_losses(self.loss_modules)

        start_epoch = self.total_epoch
        start_step = self.total_steps
        steps_per_training_epoch = len(self.model_handler.dataloader_train) // hparams.batch_size_train

        self.log_memory(hparams.use_gpu)

        t_start = timer()
        self.logger.info('Start training: {}'.format(
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        logging.info("Train parameter:\n\t" + "\n\t".join(
            sorted({"{} {}: {} {}".format(k, list(v.shape), v.requires_grad,
                                          v.device)
                    for k, v in self.model_handler.model.named_parameters()})))

        # Compute error before first iteration.
        self.trained_epochs = []
        if hparams.start_with_test:
            self.logger.info('Test epoch [{}/{}]:'.format(start_epoch, start_epoch + hparams.epochs))
            loss_dict = self.model_handler.test(hparams=hparams, total_epoch=start_epoch, total_steps=start_step,
                                                current_epoch=start_epoch)
            scheduler_loss = self.model_handler.get_summed_losses_subset(
                losses=loss_dict,
                loss_names=hparams.scheduler_loss_names)
            if np.isnan(self.best_loss) or scheduler_loss < self.best_loss:
                self.best_loss = scheduler_loss
            self.record_validation_loss(loss_dict, self.total_epoch)

        for current_epoch in range(1, hparams.epochs + 1):

            self.logger.info('Train epoch [{}/{}], step [{}/{}]:'.format(
                self.total_epoch + 1,
                start_epoch + hparams.epochs,
                self.total_steps + 1,
                start_step + hparams.epochs * steps_per_training_epoch))

            loss_dict = self.model_handler.train(hparams=hparams, total_epoch=self.total_epoch,
                                                 total_steps=self.total_steps, current_epoch=current_epoch)
            self.total_epoch += 1
            self.total_steps += steps_per_training_epoch

            if self._has_nan_loss(loss_dict):
                break

            self.record_train_loss(loss_dict, self.total_epoch)

            current_model_saved = False
            if self.total_epoch % hparams.epochs_per_test == 0:
                self.logger.info('Test epoch [{}/{}]:'.format(self.total_epoch, start_epoch + hparams.epochs))

                loss_dict = self.model_handler.test(hparams=hparams, total_epoch=self.total_epoch,
                                                    total_steps=self.total_steps, current_epoch=current_epoch)
                if self._has_nan_loss(loss_dict):
                    break

                self.record_validation_loss(loss_dict, self.total_epoch)

                scheduler_loss = self.model_handler.get_summed_losses_subset(
                    losses=loss_dict,
                    loss_names=hparams.scheduler_loss_names)

                if hparams.use_saved_learning_rate:
                    scheduler_epoch = self.total_epoch
                else:
                    scheduler_epoch = current_epoch
                assert scheduler_epoch is not None
                self.model_handler.run_scheduler(hparams=hparams,
                                                 loss=scheduler_loss,
                                                 current_epoch=scheduler_epoch)

                if hparams.out_dir is not None:
                    if np.isnan(self.best_loss) or scheduler_loss < self.best_loss:
                        self.best_loss = scheduler_loss
                        self.save_checkpoint(hparams=hparams,
                                             save_as_best_model=True)
                        current_model_saved = True

                    if hparams.epochs_per_checkpoint > 0 \
                            and self.total_epoch % hparams.epochs_per_checkpoint == 0:
                        self.save_checkpoint(hparams=hparams)
                        current_model_saved = True

            if hparams.out_dir is not None and hparams.load_newest_checkpoint \
                    and not current_model_saved:
                self.save_checkpoint(hparams=hparams,
                                     save_as_last_model=True)

        t_training = timer() - t_start
        self.logger.info('Training time: ' + str(timedelta(seconds=t_training)))
        self.log_losses(start_epoch=start_epoch)

        if hparams.out_dir is not None:
            # Check if best model should be used as final model.
            # Only possible when it was saved in out_dir.
            if hparams.use_best_as_final_model:
                self.load_best_model(hparams)

            if hparams.save_final_model:
                self.save_checkpoint(hparams)

        return (*self.get_losses(), self.model_handler)

    def sanity_check_train(self, hparams):
        assert self.model_handler is not None, "The init function has not been called before training."

        hparams.verify()  # Verify that attributes were added correctly, print warning for wrongly initialized ones.

        # Some sanity checks.
        if hparams.epochs_per_scheduler_step:
            if hparams.epochs_per_test > hparams.epochs_per_scheduler_step:
                self.logger.warning("Model is validated only every {} epochs, ".format(hparams.epochs_per_test) +
                                    "but scheduler is supposed to run every {} epochs.".format(
                                        hparams.epochs_per_scheduler_step))
            if hparams.epochs_per_test % hparams.epochs_per_scheduler_step != 0:
                self.logger.warning("hparams.epochs_per_test ({}) % hparams.epochs_per_scheduler_step ({}) != 0. "
                                    .format(hparams.epochs_per_test, hparams.epochs_per_scheduler_step) +
                                    "Note that the scheduler is only run when current_epoch % " +
                                    "hparams.epochs_per_scheduler_step == 0. Therefore hparams.epochs_per_scheduler_step " +
                                    "should be a factor of hparams.epochs_per_test.")

    def log_validation_set(self):
        if self.id_list_val is not None:
            sorted_keys = sorted(self.id_list_val)
            self.logger.info("Validation set ({}): {}".format(
                len(self.id_list_val), self.id_list_to_str(sorted_keys)))

    def log_test_set(self):
        if self.id_list_test is not None:
            sorted_keys = sorted(self.id_list_test)
            self.logger.info("Test set ({}): {}".format(
                len(sorted_keys), self.id_list_to_str(sorted_keys)))

    @staticmethod
    def id_list_to_str(id_list):
        return " ".join([os.path.join(os.path.split(os.path.dirname(id_name))[-1],
                                      os.path.splitext(os.path.basename(id_name))[0]) for id_name in id_list])

    @staticmethod
    def _has_nan_loss(loss_dict):
        for key, loss in loss_dict.items():
            if np.isnan(loss).any():
                return True
        return False

    def record_train_loss(self, loss_dict: Dict, epoch: int):
        self.train_losses.append((loss_dict, epoch))

    def record_validation_loss(self, loss_dict: Dict, epoch: int):
        self.validation_losses.append((loss_dict, epoch))

    def _get_loss_names(self):
        if len(self.train_losses) > 0:
            return list(self.train_losses[0][0].keys())
        elif len(self.validation_losses) > 0:
            return list(self.validation_losses[0][0].keys())
        else:
            return None

    def log_losses(self, start_epoch: int = -1):
        loss_names = self._get_loss_names()
        if loss_names is None:
            return

        for loss_name in loss_names:
            train_losses = np.array([loss[loss_name] for loss, epoch in self.train_losses
                                     if epoch >= start_epoch])
            validation_losses = np.array([loss[loss_name] for loss, epoch in self.validation_losses
                                          if epoch >= start_epoch])
            logging.info('Loss {} validation progress: '.format(loss_name)
                         + ', '.join('{:.4f}'.format(loss) for loss in validation_losses))
            logging.info('Loss {} train progress: '.format(loss_name)
                         + ', '.join('{:.4f}'.format(loss) for loss in train_losses))

    def get_losses(self, start_epoch: int = -1):
        loss_names = self._get_loss_names()
        if loss_names is None:
            return

        train_loss_dict = {}
        validation_loss_dict = {}
        for loss_name in loss_names:
            train_losses = np.array([loss[loss_name] for loss, epoch in self.train_losses
                                     if epoch >= start_epoch])
            validation_losses = np.array([loss[loss_name] for loss, epoch in self.validation_losses
                                          if epoch >= start_epoch])
            train_loss_dict[loss_name] = train_losses
            validation_loss_dict[loss_name] = validation_losses

        return validation_loss_dict, train_loss_dict

    def test(self, hparams):
        self.model_handler.set_dataset(hparams, self.dataset_train, self.dataset_val, self.batch_collate_fn)
        self.model_handler.set_losses(self.loss_modules)
        self.log_validation_set()
        loss_dict = self.model_handler.test(hparams=hparams, total_epoch=self.total_epoch, total_steps=self.total_steps,
                                            current_epoch=self.total_epoch)
        self.logger.info('\n\t' + '\n\t'.join('Loss {} validation: {:.4f}'.format(
            loss_name, loss_value) for loss_name, loss_value in loss_dict.items()))
        return loss_dict

    def forward(self, hparams: ExtendedHParams, ids_input: Union[str, List[str], Tuple[str, ...], os.PathLike],
                post_processing_mapping: Dict[str, str]):
        """
        Forward all given ids through the network in batches of hparams.batch_size_val.

        :param hparams:        Hyper-parameter container.
        :param ids_input:      Can be full path to file with ids, list of ids, or one id.or None.
        :return:               (Dictionary of network outputs, dictionary of post-processed (by self.OutputGen) network outputs)
        """
        assert self.model_handler is not None, "trainer.init() called before?"
        id_list = self._input_to_str_list(ids_input)

        self.logger.info("Start forwarding [{0}]".format(", ".join(str(i) for i in id_list)))
        t_start = timer()
        model_output, model_output_post = self._forward_batched(
            batch_size=hparams.batch_size_val,
            hparams=hparams,
            id_list=id_list,
            post_processing_mapping=post_processing_mapping,
            benchmark=False,
            gen_figure=False,
            synth=False)
        t_training = timer() - t_start
        self.logger.info('Forwarding time for {} sample(s): {}'.format(len(id_list), timedelta(seconds=t_training)))

        return model_output, model_output_post

    def synth(self,
              hparams: ExtendedHParams,
              ids_input: Union[str, List[str], Tuple[str, ...], os.PathLike],
              post_processing_mapping: Dict[str, str],
              plotter_configs: List[DataPlotter.Config] = None):
        """
        Synthesise all given ids with the self.synthesize function.

        :param hparams:        Hyper-parameter container.
        :param ids_input:      Can be full path to file with ids, list of ids, or one id.
        :return:               (Dictionary of network outputs, dictionary of post-processed (by self.OutputGen) network outputs)
        """

        assert self.model_handler is not None, "trainer.init() called before?"
        id_list = self._input_to_str_list(ids_input)

        self.logger.info("Start synthesising [{0}]".format(
            ", ".join(str(i) for i in id_list)))
        t_start = timer()
        model_output, model_output_post = self._forward_batched(
            batch_size=hparams.batch_size_synth,
            hparams=hparams,
            id_list=id_list,
            post_processing_mapping=post_processing_mapping,
            plotter_configs=plotter_configs,
            benchmark=False,
            gen_figure=hparams.synth_gen_figure,
            synth=True)
        t_training = timer() - t_start
        self.logger.info('Synthesis time for {} sample(s): {}'.format(
            len(id_list), timedelta(seconds=t_training)))

        return model_output, model_output_post

    def gen_figure(self,
                   hparams: ExtendedHParams,
                   ids_input: Union[str, List[str], Tuple[str, ...], os.PathLike],
                   post_processing_mapping: Dict[str, str],
                   plotter_configs: List[DataPlotter.Config]):
        """
        Generate figures for all given ids with the self.gen_figure_from_output function (has to be implemented).

        :param hparams:        Hyper-parameter container.
        :param ids_input:      Can be full path to file with ids, list of ids, or one id.
        :return:               (Dictionary of network outputs, dictionary of post-processed (by self.OutputGen) network outputs)
        """

        assert self.model_handler is not None, "trainer.init() called before?"
        id_list = self._input_to_str_list(ids_input)

        self.logger.info("Start generating figures for [{0}]".format(
            ", ".join(str(i) for i in id_list)))
        t_start = timer()
        model_output, model_output_post = self._forward_batched(
            batch_size=hparams.batch_size_gen_figure,
            hparams=hparams,
            id_list=id_list,
            post_processing_mapping=post_processing_mapping,
            plotter_configs=plotter_configs,
            benchmark=False,
            gen_figure=True,
            synth=False)
        t_training = timer() - t_start
        self.logger.info('Figure gen. time for {} sample(s): {}'.format(
            len(id_list), timedelta(seconds=t_training)))

        return model_output, model_output_post

    def benchmark(self, hparams: ExtendedHParams, post_processing_mapping: Dict[str, str],
                  ids_input: Union[str, List[str], Tuple[str, ...], os.PathLike] = None):
        """
        Benchmark the currently loaded model using the self.compute_score function (has to be implemented).

        :param hparams:        Hyper-parameter container.
        :param ids_input:      Can be full path to file with ids, list of ids, one id, or None.
                               If ids_inputs=None benchmark on test set if not None, otherwise on validation set.
        :return:               Score(s).
        """

        assert callable(getattr(self, 'compute_score', None)), "Function has to be implemented for this trainer."
        assert self.model_handler is not None, "trainer.init() called before?"

        # Select test or validation set when ids are not given explicitly.
        if ids_input is None:
            if self.id_list_test is not None and len(self.id_list_test) > 0:
                id_list = sorted(self.id_list_test)
                self.logger.info("Start benchmark on test set ({}): [{}]"
                                 .format(len(id_list), ", ".join(str(i) for i in id_list)))
            elif self.id_list_val is not None and len(self.id_list_val) > 0:
                id_list = sorted(self.id_list_val)
                self.logger.info("Start benchmark on validation set ({}): [{}]"
                                 .format(len(id_list), ", ".join(str(i) for i in id_list)))
            else:
                raise ValueError("No id list can be selected for benchmark, because non was given as parameter "
                                 "and test and validation set are empty.")
        else:
            id_list = self._input_to_str_list(ids_input)
            self.logger.info("Start benchmark on given input ({}): [{}]"
                             .format(len(id_list), ", ".join(str(i) for i in id_list)))

        t_start = timer()
        model_scores = self._forward_batched(
            batch_size=hparams.batch_size_benchmark,
            hparams=hparams,
            id_list=id_list,
            post_processing_mapping=post_processing_mapping,
            benchmark=True,
            gen_figure=False,
            synth=False)
        t_training = timer() - t_start
        self.logger.info('Benchmark time for {} sample(s): {}'.format(len(id_list), timedelta(seconds=t_training)))

        return model_scores

    def gen_output(self, hparams: ExtendedHParams, ids_input: Union[str, List[str], Tuple[str, ...], os.PathLike],
                   post_processing_mapping: Dict[str, str]):
        """
        Generate figures for all given ids with the self.gen_figure_from_output function (has to be implemented).

        :param hparams:        Hyper-parameter container.
        :param ids_input:      Can be full path to file with ids, list of ids, or one id.
        :return:               (Dictionary of network outputs, dictionary of post-processed (by self.OutputGen) network outputs)
        """

        assert self.model_handler is not None, "trainer.init() called before?"
        assert self.OutputGen is not None, "No output generator exists that can save the output."
        assert hasattr(self.OutputGen, "save_output") and callable(self.OutputGen.save_output),\
               "The output generator doesn't have a callable save_output function."
        id_list = self._input_to_str_list(ids_input)

        self.logger.info("Start generating output for [{0}]".format(", ".join(str(i) for i in id_list)))
        t_start = timer()
        model_output, model_output_post = self._forward_batched(
            batch_size=hparams.batch_size_gen_figure,
            hparams=hparams,
            id_list=id_list,
            post_processing_mapping=post_processing_mapping,
            benchmark=False,
            gen_figure=False,
            synth=False)

        for key, output in model_output_post.items():
            self.OutputGen.save_output(output, hparams.save_output_dir, key)

        t_training = timer() - t_start
        self.logger.info('Output gen. time for {} sample(s): {}'.format(len(id_list), timedelta(seconds=t_training)))

        return model_output, model_output_post

    @staticmethod
    def _input_to_str_list(input):
        # Checks for string input first.
        if isinstance(input, str):
            # Check if string is a path by trying to read ids from file.
            try:
                with open(input) as f:
                    id_list = f.readlines()
                # Trim entries in-place.
                id_list[:] = [s.strip(' \t\n\r') for s in id_list]
                return id_list
            except IOError:
                # String is single input id, convert to list.
                return [input]
        # Checks for list or tuple.
        elif isinstance(input, (list, tuple)):
            # Ensure elements are strings.
            return list(map(str, input))
        raise ValueError("Unknown input {} of type {}.".format(input,
                                                               type(input)))

    def _forward_batched(self, batch_size: int, hparams: ExtendedHParams,
                         id_list: List[str],
                         post_processing_mapping: Dict[str, str],
                         plotter_configs: List[DataPlotter.Config] = None,
                         load_target: bool = True,
                         synth: bool = False,
                         benchmark: bool = False,
                         gen_figure=False):
        assert len(id_list) > 0, "Received empty id_list."
        self.logger.info("Get model outputs as batches of size {}.".format(
            min(batch_size, len(id_list))))

        dataset = self.get_dataset(
            id_list=id_list,
            datareaders=set(self.datareaders.values()),
            hparams=hparams)
        if hparams.use_gpu:
            num_workers = hparams.dataset_num_workers_gpu
        else:
            num_workers = hparams.dataset_num_workers_cpu
        dataloader = self.model_handler._get_dataloader(
            batch_size=batch_size,
            dataset=dataset,
            batch_first=hparams.batch_first,
            common_divisor=hparams.num_gpus,
            collate_fn=self.batch_collate_fn,
            num_workers=num_workers,
            pin_memory=hparams.dataset_pin_memory,
            shuffle=False)

        dict_outputs = {}
        dict_outputs_post = {}
        for data, seq_lengths in dataloader:
            id_sub_list = data["_id_list"]
            data, seq_lengths = self.model_handler.inference(
                data=data, hparams=hparams, seq_lengths=seq_lengths)

            outputs = self.batch_decollate_fn(data, seq_lengths,
                                              batch_first=hparams.batch_first)

            for idx, id_name in enumerate(id_sub_list):
                output = {k: v[idx] for k, v in outputs.items()}
                dict_outputs[id_name] = output

                output_post = {}
                for feature_name, features in output.items():
                    if post_processing_mapping is not None \
                            and feature_name in post_processing_mapping:
                        if post_processing_mapping[feature_name] is not None:
                            reader = dataset.get_datareader_by_name(
                                post_processing_mapping[feature_name])
                            features = reader.postprocess_sample(features)
                        output_post[feature_name] = features
                    # else:
                    #     output_post[feature_name] = features
                dict_outputs_post[id_name] = output_post

                if gen_figure:
                    self.gen_figure_from_output(id_name=id_name,
                                                label_dict=output,
                                                label_post_dict=output_post,
                                                hparams=hparams,
                                                plotter_configs=plotter_configs)

        if benchmark:
            # Implementation of compute_score is checked in benchmark function.
            return self.compute_score(data=dict_outputs_post,  # TODO: Change to data_post and data.
                                      output=dict_outputs,
                                      hparams=hparams)
        if synth:
            self.gen_waveform(data=dict_outputs_post, hparams=hparams,
                              id_list=id_list)

        return dict_outputs, dict_outputs_post

    def gen_figure_from_output(self,
                               id_name: str,
                               label_dict: Dict[str, object],
                               label_post_dict: Dict[str, object],
                               hparams: ExtendedHParams,
                               plotter_configs: List[DataPlotter.Config]):

        # num_plotters = len(set(map(lambda c: c.plotter_name, plotter_configs)))

        with ExitStack() as stack:  # Open a variable number of context managers.
            plotters = {}
            for plotter_conf in plotter_configs:
                if plotter_conf.plotter_name not in plotters:
                    # plotter = DataPlotter()
                    plotter = stack.enter_context(DataPlotter())
                    net_name = os.path.basename(hparams.model_name)
                    plotter.set_title(id_name + ' - ' + net_name)
                    plotters[plotter_conf.plotter_name] = plotter

                if plotter_conf.post_processed:
                    features = label_post_dict[plotter_conf.feature_name]
                else:
                    features = label_dict[plotter_conf.feature_name]

                plotter_conf.plot_fn(
                    plotter=plotters[plotter_conf.plotter_name],
                    grid_indices=plotter_conf.grid_indices,
                    id_name=id_name,
                    features=features,
                    hparams=hparams,
                    plotter_config=plotter_conf,
                    label_dict=label_dict,
                    label_post_dict=label_post_dict)
                if plotter_conf.annotation_fn is not None:
                    plotter_conf.annotation_fn(plotter, id_name, hparams)

            figure_dir = self._get_figure_dir(hparams)
            for plotter_name, plotter in plotters.items():
                plotter.gen_plot()
                filename = "{}_e{}{}{}".format(
                    id_name,
                    self.total_epoch,  # TODO: Is self.total_epoch always defined?
                    plotter_name if plotter_name != "default" else "",
                    hparams.gen_figure_ext)
                plotter.save_to_file(os.path.join(figure_dir, filename))

    @staticmethod
    def _get_figure_dir(hparams: ExtendedHParams):
        return os.path.join(hparams.out_dir, hparams.model_name, "figures")

    @staticmethod
    def plot1d(
            plotter: DataPlotter,
            plotter_config: DataPlotter.Config,
            grid_indices: List[int],
            id_name: str,
            features: np.ndarray,
            feature_name: str = None,
            labels: Tuple[str, str] = (None, None),
            xlim: Union[str, Tuple[float, float]] = (None, None),
            ylim: Union[str, Tuple[float, float]] = (None, None),
            # linewidth: float = None,
            # colour: str = None,
            # alpha: float = None
            *args,
            **kwargs):

        if grid_indices is None:
            grid_idx = plotter.get_next_free_grid_idx()
        else:
            grid_idx = grid_indices[0]

        if feature_name is not None:
            data_list = [(features, feature_name)]
        else:
            data_list = [(features,)]
        plotter.set_data_list(grid_idx=grid_idx, data_list=data_list)

        plotter.set_label(grid_idx=grid_idx, xlabel=labels[0], ylabel=labels[1])

        xlim = ModularTrainer._get_lims(axis=1, features=features, lims=xlim)
        ylim = ModularTrainer._get_lims(axis=0, features=features, lims=ylim)
        plotter.set_lim(grid_idx=grid_idx,
                        xmin=xlim[0], xmax=xlim[1],
                        ymin=ylim[0], ymax=ylim[1])

    @staticmethod
    def plot_specshow(
            plotter: DataPlotter,
            plotter_config: DataPlotter.Config,
            grid_indices: List[int],
            id_name: str,
            features: np.ndarray,
            spec_slice: slice = None,
            labels: Tuple[str, str] = (None, None),
            xlim: Union[str, Tuple[float, float]] = (None, None),
            ylim: Union[str, Tuple[float, float]] = (None, None),
            *args,
            **kwargs):

        if grid_indices is None:
            grid_idx = plotter.get_next_free_grid_idx()
        else:
            grid_idx = grid_indices[0]

        if spec_slice is not None:
            features = features[spec_slice]
        plotter.set_specshow(grid_idx=grid_idx, spec=features)

        plotter.set_label(grid_idx=grid_idx, xlabel=labels[0], ylabel=labels[1])

        xlim = ModularTrainer._get_lims(axis=1, features=features, lims=xlim)
        ylim = ModularTrainer._get_lims(axis=0, features=features, lims=ylim)
        plotter.set_lim(grid_idx=grid_idx,
                        xmin=xlim[0], xmax=xlim[1],
                        ymin=ylim[0], ymax=ylim[1])

    @staticmethod
    def _get_lims(axis, features, lims):
        if lims == "centred":
            max_ = abs(features).max(axis=axis) * 1.1
            return -max_, max_
        else:
            return lims

    def gen_waveform(
            self,
            id_list: List[str],
            data: Union[Dict[str, np.ndarray], Dict[str, Dict[str, np.ndarray]]],
            hparams: ExtendedHParams,
            use_model_name: bool = True,
            has_deltas: bool = False) -> None:

        if type(next(iter(data.values()))) is dict:
            if hparams.has_value("synth_feature_names"):
                feature_names = hparams.synth_feature_names
                if type(feature_names) not in [list, tuple]:
                    feature_names = (feature_names,)
            else:
                feature_names = list(next(iter(data.values())).keys())
                self.logger.warning(
                    "hparams.synth_feature_names is not defined, using all "
                    "features {} instead.".format(feature_names))

            data = {id_name: np.concatenate([features[feature_name]
                                             for feature_name in feature_names],
                                            axis=1)
                    for id_name, features in data.items()}

            self.logger.info("Synthesise from " + ", ".join(feature_names))

        # Create speaker subdirectories if necessary.
        for id_name in id_list:
            path_split = os.path.split(id_name)
            if len(path_split) > 2:
                makedirs_safe(os.path.join(hparams.synth_dir, *path_split[:-1]))

        if hparams.synth_vocoder == "WORLD":
            Synthesiser.run_world_synth(
                data, hparams, use_model_name=use_model_name,
                has_deltas=has_deltas, epoch=self.total_epoch,
                step=self.total_steps)

        elif hparams.synth_vocoder == "r9y9wavenet_mulaw_16k_world_feats_English":
            Synthesiser.run_r9y9wavenet_mulaw_world_feats_synth(
                data, hparams, epoch=self.total_epoch, step=self.total_steps)

        elif hparams.synth_vocoder == "raw":
            # The features in the data dictionary are raw waveforms and can be
            # written directly to the file.
            Synthesiser.run_raw_synth(
                data, hparams, epoch=self.total_epoch, step=self.total_steps)

        elif hparams.synth_vocoder == "80_SSRN_English_GL":
            # Use a pre-trained spectrogram super resolution network for English
            # and Griffin-Lim. The features in the data should be mfbanks.
            raise NotImplementedError()  # TODO

        elif hparams.synth_vocoder == "GL_on_log":
            Synthesiser.run_griffin_lim_on_log(
                data, hparams, epoch=self.total_epoch, step=self.total_steps)

        elif hparams.synth_vocoder == "GL":
            Synthesiser.run_griffin_lim(
                data, hparams, epoch=self.total_epoch, step=self.total_steps)

        elif hparams.synth_vocoder == "r9y9wavenet":
            # Synthesise with a pre-trained r9y9 WaveNet. The hyper-parameters
            # have to match the model.
            Synthesiser.run_wavenet_vocoder(
                data, hparams, epoch=self.total_epoch, step=self.total_steps)

        # elif hparams.synth_vocoder == "STRAIGHT":  # Add more vocoders here.

        else:
            raise NotImplementedError("Unknown vocoder type {}."
                                      .format(hparams.synth_vocoder))

    def get_labels(self, reader_name, id_name):
        dataset = PyTorchDatareadersDataset(id_list=None, datareaders=set(
            self.datareaders.values()))
        reader = dataset.get_datareader_by_name(reader_name)
        return reader[id_name]

    def copy_synth(self, hparams, id_list):
        assert hparams.has_value("synth_feature_names"), \
            "hparams.synth_feature_names has to be given."

        feature_names = hparams.synth_feature_names
        if type(feature_names) not in [list, tuple]:
            feature_names = (feature_names,)

        ids_input = self._input_to_str_list(id_list)
        hparams = copy.deepcopy(hparams)

        dataset = PyTorchDatareadersDataset(id_list=None, datareaders=set(
            self.datareaders.values()))
        readers = [dataset.get_datareader_by_output_name(name) for name
                   in feature_names]
        data = {}
        for id_name in id_list:
            # features = np.concatenate([reader[id_name] for reader in readers], axis=1)
            # features = np.concatenate([reader.postprocess_sample(reader[id_name]) for reader in readers], axis=1)
            features_list = [reader.load(id_name) for reader in readers]
            features = np.concatenate(features_list, axis=1)
            data[id_name] = features

        old_synth_file_suffix = hparams.synth_file_suffix
        hparams.synth_file_suffix += "_ref"
        self.gen_waveform(ids_input, data, hparams, use_model_name=False)
        hparams.synth_file_suffix = old_synth_file_suffix
