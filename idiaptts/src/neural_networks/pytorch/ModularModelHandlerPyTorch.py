#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#
import platform
import os
import logging
import resource
from datetime import datetime
from operator import itemgetter
import glob
import re
from functools import partial

import numpy as np
import random
from typing import Union, Any, List, Optional, cast, Tuple

# Third-party imports.
import torch
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader
from torch.nn import DataParallel
from typing import Union, Any, List, Optional, cast, Dict
from torch.nn.utils.rnn import pad_sequence
import jsonpickle

# Local source tree imports.
from idiaptts.src.ExtendedHParams import ExtendedHParams
from idiaptts.misc.utils import log_git_hash
# from idiaptts.src.neural_networks.pytorch.models.EncDecDyn import EncDecDyn
# from idiaptts.src.neural_networks.pytorch.ModelHandlerPyTorch import ModelHandlerPyTorch
from idiaptts.misc.utils import get_gpu_memory_map
from idiaptts.src.neural_networks.ModelHandler import ModelHandler
from idiaptts.misc.utils import makedirs_safe, local_modification_time
from idiaptts.src.neural_networks.pytorch.ExponentialMovingAverage import ExponentialMovingAverage
from idiaptts.src.neural_networks.pytorch.ExtendedExponentialLR import ExtendedExponentialLR


class ModularModelHandlerPyTorch(ModelHandler):
    """
    Provides functionality to work with multiple network architectures. It allows to create, load and save a model,
    train and test it and load the data for it. This class creates a wrapper around the framework used to implement
    the models. This class should be implemented for each framework, this is the class for PyTorch.
    """
    logger = logging.getLogger(__name__)

    def __init__(self):
        super().__init__()

        self.optimiser = None
        self.scheduler = None
        self._scheduler_step_fn = None

        self.ema = None

    @staticmethod
    def cuda_is_available():
        return torch.cuda.is_available()

    @staticmethod
    def device_count():
        return torch.cuda.device_count()

    @staticmethod
    def seed(seed):
        torch.manual_seed(seed)

    def save_checkpoint(self,
                        model_path: Union[str, os.PathLike],
                        best_loss: np.ndarray = np.inf,
                        epoch: int = None, step: int = None,
                        save_as_best_model: bool = False,
                        save_as_epoch: bool = True,
                        save_as_last_model: bool = False,
                        save_as_step: bool = True):
        assert save_as_best_model or save_as_last_model or step is not None \
            or epoch is not None, "Epoch or step needs to be given."
        assert model_path is not None, "Given model_path cannot be None."

        if save_as_best_model:
            suffix = "best"
        elif save_as_last_model:
            suffix = "last"
        elif epoch is not None and save_as_epoch:
            suffix = "e{}".format(epoch)
        elif step is not None and save_as_step:
            suffix = "s{}".format(step)
        else:
            raise NotImplementedError()
        self.logger.info("Save {} checkpoint to {}.".format(suffix, model_path))
        makedirs_safe(model_path)

        config_json = self.model.get_config_as_json()
        # TODO: Dump hparams in it as well?
        with open(os.path.join(model_path, "config.json"), "w") as f:
            f.write(config_json)

        params = self.model.state_dict()
        if self.ema:
            # Update only the parameters which are shadowed.
            params.update(self.ema.shadow)
            self.logger.info("Updated checkpoint with EMA model parameters {}."
                             .format(", ".join(self.ema.shadow.keys())))
        else:
            params = self.model.state_dict()

        checkpoint = {"params": params, "epoch": epoch, "step": step}
        torch.save(checkpoint, os.path.join(model_path, "params_" + suffix))

        if self.optimiser is not None:
            opt_params = self.optimiser.state_dict()
            checkpoint = {"params": opt_params, "epoch": epoch, "step": step,
                          "best_loss": best_loss}
            torch.save(checkpoint, os.path.join(model_path, "optimiser_" + suffix))

        if self.scheduler is not None:
            scheduler_params = self.scheduler.state_dict()
            checkpoint = {"params": scheduler_params, "epoch": epoch,
                          "step": step}
            torch.save(checkpoint, os.path.join(model_path, "scheduler_" + suffix))

    def load_checkpoint(self, hparams: ExtendedHParams,
                        model_path: Union[str, os.PathLike], epoch: int = None,
                        ignore_layers: bool = True, load_optimiser: bool = True,
                        load_scheduler: bool = True, step: int = None,
                        verbose: bool = True, load_best_model: bool = False):
        """
        Load a trainer and model from    save_as_best_model: bool = False): a checkpoint.

        :param hparams: Hyper-parameter container
        :type hparams: ExtendedHParams
        :param model_path: Path to folder with save files of the checkpoint (config.json, params_*, trainer_*)
        :type model_path: String or Path
        :param epoch: Epoch of the checkpoint to load, use -1 to load best model, defaults to None
        :type epoch: int, optional
        :param ignore_layers: Whether to ignore layers specified in hparams, defaults to True
        :type ignore_layers: bool, optional
        :param load_optimiser: Whether to load the optimiser state, defaults to True
        :type load_optimiser: bool, optional
        :param step: Step of the checkpoint to load, use -1 to load best model, defaults to None
        :type step: int, optional
        :param verbose: Additional logging of checkpoint creation time, defaults to True
        :type verbose: bool, optional
        :param load_best_model: If true, epoch and step are ignored and the best model is loaded, defaults to False
        :type load_best_model: bool, optional
        :return: (best_loss, epoch, step) tuple of loaded checkpoint
        :type: Tuple[float, int, int]
        """
        assert load_best_model or step is None or epoch is None, \
            "Only epoch ({}) OR step ({}) can be not None".format(epoch, step)

        if load_best_model or epoch == -1 or step == -1:
            suffix = "_best"
        elif hparams.load_newest_checkpoint:
            assert step is None and epoch is None, \
                "epoch ({}) and step ({}) need to be None when loading newest "\
                "model.".format(epoch, step)

            file_list = glob.glob(os.path.join(model_path, "params_*"))
            if len(file_list) == 0:
                raise FileNotFoundError("No newest checkpoint found in {}."
                                        .format(model_path))
            elif len(file_list) == 1:
                latest_params = file_list[0]
            else:
                file_list = [f for f in file_list if os.path.basename(f) not in
                             ["params_e0", "params_s0"]]  # Ignore initial state
                latest_params = max(file_list, key=os.path.getctime)
            suffix = "_" + os.path.basename(latest_params).split('_')[1]
        else:
            assert load_best_model or step is not None or epoch is not None, \
                "Either step or epoch is required. Use -1 in one of them to " \
                "load the best model."
            if step is not None:
                suffix = "_s{}".format(step)
            else:
                suffix = "_e{}".format(epoch)
        params_path = os.path.join(model_path, "params" + suffix)

        if verbose:
            mod_time = local_modification_time(params_path)
            message = "Load model state dict from {} (last modified {})".format(
                params_path, mod_time)
            if ignore_layers and hparams.ignore_layers is not None \
                    and len(hparams.ignore_layers) > 0:
                message += " ignoring {}".format(hparams.ignore_layers)
            self.logger.info(message)

        checkpoint = torch.load(params_path, map_location=lambda storage,
                                loc: storage)
        try:
            params = checkpoint["params"]
        except KeyError:
            params = checkpoint["model_state_dict"]  # Legacy support

        best_loss = np.inf
        epoch = checkpoint["epoch"]
        step = checkpoint["step"] if "step" in checkpoint else None
        self.logger.info("Load {}{}".format(
            "epoch {}, ".format(epoch) if epoch is not None else "",
            "step {}".format(step) if step is not None else ""))

        if self.model is None:
            with open(os.path.join(model_path, "config.json"), "r") as f:
                json_str = f.read()
            config_json = jsonpickle.decode(json_str)
            self.model = config_json.create_model()

        if hparams.has_value("layer_map") and len(hparams.layer_map) > 0:
            params = self._map_layer_names(params, hparams.layer_map, verbose)

        if ignore_layers:
            params = self._remove_ignored_layers(params, self.model, hparams)
        missing_keys, unexpected_keys = self.model.load_state_dict(
            params, strict=not hparams.allow_missing_layers)
        if verbose:
            if len(missing_keys) > 0:
                self.logger.warning("Did not load: {}".format(
                    ", ".join(missing_keys)))
            if len(unexpected_keys) > 0:
                self.logger.warning("Found unexpected keys: {}".format(
                    ", ".join(unexpected_keys)))

        if load_optimiser:
            opt_params_path = os.path.join(model_path, "optimiser" + suffix)
            checkpoint = torch.load(opt_params_path, map_location=lambda storage,
                                    loc: storage)
            if "best_loss" in checkpoint and (not ignore_layers
                                              or hparams.ignore_layers is None
                                              or len(hparams.ignore_layers) == 0):
                best_loss = checkpoint["best_loss"]
            opt_params = checkpoint["params"]
            # if opt_params is not None:
            self._load_optimiser(opt_params, hparams)

            if load_scheduler:
                scheduler_params_path = os.path.join(model_path,
                                                     "scheduler" + suffix)
                if os.path.isfile(scheduler_params_path):
                    checkpoint = torch.load(opt_params_path,
                                            map_location=lambda storage,
                                            loc: storage)
                    scheduler_params = checkpoint["params"]
                    self._load_scheduler(
                        scheduler_params,
                        epoch if epoch is not None else checkpoint['epoch'],
                        step if step is not None else checkpoint['step'],
                        hparams)

        if hparams.use_gpu:
            if hasattr(self.model, "set_gpu_flag") \
                    and callable(self.model.set_gpu_flag):
                self.model.set_gpu_flag(hparams.use_gpu)
            self.model = self.model.cuda()

            if self.optimiser is not None:
                self._optimiser_to_gpu()

        return best_loss, epoch, step

    @staticmethod
    def _map_layer_names(params, layer_map, verbose=True):
        new_params = {}
        for param_name, param in params.items():
            new_name = ModularModelHandlerPyTorch._map_layer_name(
                param_name, layer_map, verbose)
            new_params[new_name] = param
        return new_params

    @staticmethod
    def _map_layer_name(param_name, layer_map, verbose=True):
        for pattern, replacement in layer_map:
            match = re.search(pattern, param_name)
            if match is not None:
                new_name = re.sub(pattern, replacement, param_name)
                if verbose:
                    logging.info("Found layer mapping: {} ==> {}".format(
                        param_name, new_name))
                return new_name
        return param_name

    @staticmethod
    def _remove_ignored_layers(model_dict, model, hparams):
        keys_to_pop = []
        if hasattr(hparams, "ignore_layers") and len(hparams.ignore_layers) > 0:
            for ignored_layer in hparams.ignore_layers:
                found_key = False
                for key in model_dict.keys():
                    if re.match(ignored_layer, key):
                        found_key = True
                        keys_to_pop.append(key)
                if not found_key:
                    raise KeyError("Cannot find layer {} in saved model dict: {}"
                                   .format(ignored_layer,
                                           ", ".join(model_dict.keys())))

            model_dict = {k: v for k, v in model_dict.items()
                          if k not in keys_to_pop}
            org_dict = model.state_dict()
            for k in keys_to_pop:
                # Substitute the ignored layers with the current layer parameters.
                if k in org_dict:
                    model_dict[k] = org_dict[k]
        if len(keys_to_pop) > 0:
            logging.info("Ignored {}".format(keys_to_pop))
        return model_dict

    def _load_optimiser(self, opt_params, hparams):
        self.set_optimiser(hparams, reset=True)
        try:
            self.optimiser.load_state_dict(opt_params)
        except ValueError as e:
            self.logger.warning(
                "State dict for optimiser {} miss matches checkpoint's optimiser"
                " state dict: {}\nContinuing without loading optimiser instead."
                .format(hparams.optimiser_type, e))

        if self.optimiser is not None:
            # Initial learning rate is required by some optimisers to compute
            # the learning rate of the current epoch/step.
            self._update_optimiser_initial_lr(hparams)

    def _update_optimiser_initial_lr(self, hparams: ExtendedHParams):
        # Handle the case where multiple parameter groups are given.
        if "params" in hparams.optimiser_args:
            for group, opt_args in zip(self.optimiser.param_groups,
                                       hparams.optimiser_args["params"]):
                group.setdefault('initial_lr', opt_args["lr"])
        else:
            if "lr" in hparams.optimiser_args:
                initial_lr = hparams.optimiser_args["lr"]
            elif hasattr(hparams, "learning_rate"):  # Legacy support
                initial_lr = hparams.learning_rate
            else:
                return

            for group in self.optimiser.param_groups:
                # if hasattr(group, 'initial_lr'):
                group.setdefault('initial_lr', initial_lr)

    def _load_scheduler(self, scheduler_params, current_epoch, current_step, hparams):
        if hparams.epochs_per_scheduler_step is not None:
            self.set_scheduler(hparams, current_epoch=current_epoch, reset=True)
        elif hparams.iterations_per_scheduler_step is not None:
            self.set_scheduler(hparams, current_step=current_step, reset=True)
        else:
            self.set_scheduler(hparams, current_epoch=current_epoch, current_step=current_step, reset=True)

        if self.scheduler is not None:
            try:
                self.scheduler.load_state_dict(scheduler_params)
            except ValueError as e:
                self.logger.warning("State dict for scheduler {} miss matches checkpoint's scheduler "
                                    "state dict: {}\nContinuing without loading scheduler instead."
                                    .format(hparams.scheduler_type, e))

    def _optimiser_to_gpu(self):
        # TODO: Is this still necessary in PyTorch > 1.7
        # self.optimiser.cuda()  # Not implemented in master, but here:
        # https://github.com/andreh7/pytorch/blob/235ce5ba688a49f804422226ddc62a721bb811e0/torch/optim/optimizer.py
        # Requires the following function.
        def _transform_state(optimiser, transformation, filter_func, state=None):
            """Applies ``transformation`` recursively to each item in ``self.state`` for which ``filter_func`` returns True.
            Arguments:
                transformation (item -> item): function to be applied to each item passing ``filter_func``.
                filter_func (item -> `bool`): function which must return True for each item to which ``transformation`` should be applied.
            """
            if state is None:
                state = optimiser.state
            for key, value in state.items():
                if isinstance(value, dict):
                    _transform_state(optimiser, transformation, filter_func, value)
                else:
                    if filter_func(value):
                        state[key] = transformation(value)

        _transform_state(self.optimiser, lambda t: t.cuda(), lambda t: torch.is_tensor(t))  # Doing it manually.

    def create_model(self, model_config, use_gpu):
        self.logger.info("Create network from config: {}".format(type(model_config)))
        self.model = model_config.create_model()
        if use_gpu:
            self.model = self.model.cuda()

    @staticmethod
    def prepare_batch(batch, common_divisor=1, batch_first=False):
        # Remove samples if not equally dividable by given divisor (# GPUs).
        # Remove before sorting to keep it unbiased.
        assert (len(batch) >= common_divisor)
        remainder = len(batch) % common_divisor
        if remainder > 0:
            batch = batch[:-remainder]

        # Remove dataset.
        dataset = batch[0][1]
        batch = [b[0] for b in batch]

        # Concat same features.
        data = dict()
        lengths = dict()
        for key in batch[0].keys():
            values = [b[key] for b in batch if key in b]
            if key != "_id_list":
                lengths[key] = torch.tensor([x.shape for x in values],
                                            dtype=torch.long)
            try:
                data_reader = dataset.get_datareader_by_output_name(key)
                max_lengths, max_lengths_indices = lengths[key].max(axis=0)
                max_frames = max_lengths[0]

                if data_reader.min_frames is not None \
                        and max_frames < data_reader.min_frames:
                    max_length_idx = max_lengths_indices[0]
                    max_value = values[max_length_idx]
                    padding = [(0, data_reader.min_frames - max_frames)]
                    padding += [(0, 0) for _ in range(max_value.ndim - 1)]

                    # Padding the longest one ensures that the whole
                    # batch will later have the min_frames length.
                    values[max_length_idx] = data_reader.pad(max_value,
                                                             padding)
                    max_lengths[0] = data_reader.min_frames
                    max_frames = max_lengths[0]

                # Check whether other but the temporal dimension need to
                # be padded (used in the fixed attention matrix).
                if data_reader.other_pad_dims is not None:
                    for idx, sample in enumerate(values):
                        padding = [(0, 0) for _ in range(sample.ndim)]
                        for dim in data_reader.other_pad_dims:
                            if dim == 0:  # Temporal dim is padded anyway.
                                continue
                            max_length_of_dim = max_lengths[dim]
                            max_lengths[dim] = max_length_of_dim
                            sample_length = lengths[key][idx][dim]
                            padding[dim] = (0, max_length_of_dim - sample_length)
                        values[idx] = data_reader.pad(sample, padding)

                if data_reader.requires_seq_mask:
                    assert data_reader.other_pad_dims is None, "Sequence mask"\
                        "for padding in multiple dimensions is not implemented."
                    mask = ModularModelHandlerPyTorch.sequence_mask(
                        lengths[key][:, 0], max_frames, batch_first=batch_first)
                    data[key + "_mask"] = mask
                    # Used by losses to compute correct mean over time.
                    lengths[key + "_mask"] = lengths[key]

                if hasattr(data_reader, "unsorted_pad_sequence") \
                        and callable(data_reader.unsorted_pad_sequence):
                    padded_values = data_reader.unsorted_pad_sequence(
                        values, batch_first, lengths[key])
                    data[key] = torch.from_numpy(padded_values).float()
                else:
                    data[key] = ModularModelHandlerPyTorch.unsorted_pad_sequence(
                        values, batch_first)
            except (ValueError, KeyError):
                data[key] = list(values)

        # The rest of the code only cares about temporal lengths.
        temporal_lengths = {k: torch.tensor([l[0] for l in ls]) for k, ls in
                            lengths.items()}
        return data, temporal_lengths

    @staticmethod
    def sequence_mask(sequence_length, max_len=None, batch_first=False):
        """Code adapted from https://github.com/r9y9/wavenet_vocoder/blob/master/train.py."""
        # Get some dimensions and sizes.
        time_dim = 1 if batch_first else 0
        if max_len is None:
            max_len = sequence_length.data.max()
        batch_size = len(sequence_length)

        # Create a range from 0 to max.
        seq_range = torch.arange(0, max_len, dtype=torch.long)
        # Expand the range to all samples in the batch.
        if batch_first:
            seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
        else:
            seq_range_expand = seq_range.unsqueeze(1).expand(max_len, batch_size)

        # Expand the lengths of all samples to the full size.
        # seq_length_expand = torch.tensor(sequence_length, dtype=seq_range_expand.dtype, device=seq_range_expand.device).unsqueeze(time_dim).expand_as(seq_range_expand)
        seq_length_expand = sequence_length.detach().clone()\
            .type(seq_range_expand.dtype).to(device=seq_range_expand.device)\
            .detach().unsqueeze(time_dim).expand_as(seq_range_expand)

        # Compare element by element and return it as float.
        return (seq_range_expand < seq_length_expand).unsqueeze(-1).contiguous().float()

    @staticmethod
    def unsorted_pad_sequence(sequence, batch_first):

        sequence = [torch.from_numpy(s) if isinstance(s, np.ndarray) else s
                    for s in sequence]

        return pad_sequence(sequence, batch_first)

    def set_dataset(self, hparams, dataset_train, dataset_val, collate_fn=None):
        common_divisor = hparams.num_gpus  # Will be 1 if used on CPU.
        collate_fn = self.prepare_batch if collate_fn is None else collate_fn
        num_workers = hparams.dataset_num_workers_gpu if hparams.use_gpu else hparams.dataset_num_workers_cpu

        # assert hparams.batch_first, "ModularModelHandlerPyTorch does not support batch_first=False"

        self.dataloader_train = self._get_dataloader(
            batch_size=hparams.batch_size_train,
            dataset=dataset_train,
            batch_first=hparams.batch_first,
            collate_fn=collate_fn,
            common_divisor=common_divisor,
            num_workers=num_workers,
            pin_memory=hparams.dataset_pin_memory,
            shuffle=hparams.shuffle_train_set)

        self.dataloader_val = self._get_dataloader(
            batch_size=hparams.batch_size_val,
            dataset=dataset_val,
            batch_first=hparams.batch_first,
            collate_fn=collate_fn,
            common_divisor=common_divisor,
            num_workers=num_workers,
            pin_memory=hparams.dataset_pin_memory,
            shuffle=hparams.shuffle_val_set)

    def _get_dataloader(self, batch_size, dataset, batch_first=True, collate_fn=None, common_divisor=1, num_workers=1,
                        pin_memory=True, shuffle=False):
        collate_fn = self.prepare_batch if collate_fn is None else collate_fn
        if isinstance(dataset, torch.utils.data.dataset.IterableDataset):
            return DataLoader(dataset=dataset,
                              batch_size=None,
                              num_workers=num_workers,
                              collate_fn=partial(collate_fn,
                                                 common_divisor=common_divisor,
                                                 batch_first=batch_first),
                              pin_memory=pin_memory,
                              worker_init_fn=dataset.worker_init_fn)
        else:
            return DataLoader(dataset=dataset,
                              batch_size=batch_size,
                              shuffle=shuffle,
                              num_workers=num_workers,
                              collate_fn=partial(collate_fn,
                                                 common_divisor=common_divisor,
                                                 batch_first=batch_first),
                              pin_memory=pin_memory)

    def set_losses(self, losses):
        self.losses = losses

    def set_optimiser(self, hparams, reset=False):
        """Initialise a PyTorch optimiser here."""
        if self.optimiser is None or reset:
            if hparams.optimiser is None:
                self.logger.info("Create {} optimiser.".format(hparams.optimiser_type))

                if "params" in hparams.optimiser_args:
                    optim_arguments = hparams.optimiser_args
                else:
                    if "lr" not in hparams.optimiser_args:  # Backwards compatible.
                        try:
                            hparams.optimiser_args["lr"] = hparams.learning_rate
                        except AttributeError:
                            raise AttributeError("Learning rate not defined in hparams.optimiser_args[\"lr\"]")
                    optim_arguments = {"params": self.model.parameters()}
                    optim_arguments.update(hparams.optimiser_args)
                # Model is new.
                if hparams.optimiser_type == "Adam":
                    self.optimiser = torch.optim.Adam(**optim_arguments)
                elif hparams.optimiser_type == "SGD":
                    self.optimiser = torch.optim.SGD(**optim_arguments)
                # TODO: Implement the others here.
                else:
                    raise NotImplementedError("Optimiser type {} is not implemented.".format(hparams.optimiser_type))
            else:
                self.optimiser = hparams.optimiser(self.model.parameters())

        # Model was loaded from checkpoint, override learning rate if requested.
        if not hparams.use_saved_learning_rate and "lr" in hparams.optimiser_args:
            for g in self.optimiser.param_groups:
                g['lr'] = hparams.optimiser_args["lr"]

    def set_scheduler(self, hparams, current_epoch=None, current_step=None, reset=False):
        """Initialise a PyTorch scheduler here."""
        if self.scheduler is not None and not reset:
            return
        if hparams.scheduler is None:
            if hparams.scheduler_type.lower() == "none":
                return
            assert hparams.scheduler_type != "default", "Please define a default scheduler type in the trainer class."

            self.logger.info("Create {} scheduler.".format(hparams.scheduler_type))

            # PyTorch schedulers use -1 as first epoch and call step immediately.
            if current_epoch == 0:
                current_epoch = -1
            if current_step == 0:
                current_step = -1

            if hparams.scheduler_type == "Plateau":
                self.scheduler = ReduceLROnPlateau(self.optimiser, **hparams.scheduler_args)
                self._scheduler_step_fn = self._scheduler_step_with_loss
                if hparams.epochs_per_scheduler_step is None and hparams.iterations_per_scheduler_step is None:
                    hparams.epochs_per_scheduler_step = 1
                return

            if current_step is None and self.dataloader_train is not None:
                current_step = max((current_epoch - 1) * len(self.dataloader_train), -1)
            if hparams.scheduler_type == "ExtendedExponentialLR":

                if hparams.epochs_per_scheduler_step is None:
                    if hparams.iterations_per_scheduler_step is None:
                        hparams.iterations_per_scheduler_step = 1
                    self.scheduler = ExtendedExponentialLR(self.optimiser, last_epoch=current_step,
                                                           **hparams.scheduler_args)
                else:
                    self.scheduler = ExtendedExponentialLR(self.optimiser, last_epoch=current_epoch - 1,
                                                           **hparams.scheduler_args)
                self._scheduler_step_fn = self._scheduler_step
                return

            if hparams.scheduler_type == "Exponential":
                if hparams.epochs_per_scheduler_step is None:
                    if hparams.iterations_per_scheduler_step is None:
                        hparams.iterations_per_scheduler_step = 1
                    self.scheduler = ExponentialLR(self.optimiser, last_epoch=current_step,
                                                   **hparams.scheduler_args)
                else:
                    self.scheduler = ExponentialLR(self.optimiser, last_epoch=current_epoch - 1,
                                                   **hparams.scheduler_args)
                self._scheduler_step_fn = self._scheduler_step
                return

            elif hparams.scheduler_type == "Noam":
                assert "wormup_steps" in hparams.scheduler_args, "Please define wormup_steps in hparams.scheduler_args."

                def noam_decay(iteration):
                    wormup_steps = float(hparams.scheduler_args['wormup_steps'])
                    return wormup_steps**0.5 * np.minimum((iteration + 1) * wormup_steps**-1.5,
                                                          (iteration + 1)**-0.5)

                if hparams.epochs_per_scheduler_step is None:
                    if hparams.iterations_per_scheduler_step is None:
                        hparams.iterations_per_scheduler_step = 1
                    self.scheduler = LambdaLR(self.optimiser, noam_decay, last_epoch=current_step)
                else:
                    self.scheduler = LambdaLR(self.optimiser, noam_decay, last_epoch=current_epoch - 1)
                self._scheduler_step_fn = self._scheduler_step
                return

            # TODO: Implement the others here.
            raise NotImplementedError("Scheduler type {} is not implemented.".format(hparams.scheduler_type))
        else:
            self.scheduler = hparams.scheduler(self.optimiser)

    def _scheduler_step_with_loss(self, loss):
        self.scheduler.step(loss)

    def _scheduler_step(self, loss):
        self.scheduler.step()

    def test(self, hparams, total_epoch, total_steps, current_epoch):
        return self.process_dataloader(dataloader=self.dataloader_val,
                                       hparams=hparams,
                                       total_epoch=total_epoch,
                                       total_steps=total_steps,
                                       current_epoch=current_epoch,
                                       training=False)

    def train(self, hparams, total_epoch, total_steps, current_epoch):
        if hparams.ema_decay and not self.ema:
            self.ema = ExponentialMovingAverage(self.model, hparams.ema_decay)

        return self.process_dataloader(dataloader=self.dataloader_train,
                                       hparams=hparams,
                                       total_epoch=total_epoch,
                                       total_steps=total_steps,
                                       current_epoch=current_epoch,
                                       training=True)

    def process_dataloader(self,
                           dataloader: DataLoader,
                           hparams: ExtendedHParams,
                           total_epoch: int,
                           total_steps: int,
                           current_epoch: int = None,
                           training: bool = True):
        if hparams.use_gpu:
            assert (hparams.num_gpus <= torch.cuda.device_count()), \
                "Specified number of GPUs is incorrect."

        try:
            from torch.utils.tensorboard import SummaryWriter

            if hparams.has_value("tensorboard_dir"):
                tensorboard_dir = hparams.tensorboard_dir
            else:
                tensorboard_dir = os.path.join(hparams.out_dir,
                                               hparams.model_name,
                                               "tensorboard")
            tb_writer = SummaryWriter(log_dir=tensorboard_dir)
        except ImportError:
            tb_writer = None

        model = self.model
        if training:
            model.train()
            msg = "{}: Train with {} on ".format(
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                self.optimiser)
            if hparams.use_gpu:
                msg += str(torch.cuda.device_count()) + " GPU(s)."
            else:
                msg += "1 CPU."
            self.logger.info(msg),
        else:
            if self.ema is not None:
                self.logger.info("Using averaged model for validation.")
                model = self.ema.model
            model.eval()
            self.logger.info("{}: Compute loss of validation set.".format(
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

        if hparams.log_memory_consumption:
            self.logger.info('CPU: {:.0f} MB, GPU: {} MB'.format(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e3,
                str(get_gpu_memory_map()) if hparams.use_gpu else "-"))

        # Multi-GPU support.
        if hparams.num_gpus > 1:
            model = DataParallel(model, dim=0 if hparams.batch_first else 1)
            # Make the init_hidden method directly accessible.
            model.init_hidden = model.module.init_hidden

        # Log loss after each <hparams.logging_batch_index_perc>% of batches.
        logging_batch_index = (len(dataloader) // hparams.logging_batch_index_perc) + 1

        total_losses = dict()

        # for params in reversed(list(self.model.parameters())):
        #         params.retain_grad()

        for batch_index, batch in enumerate(dataloader):

            if hparams.use_gpu:
                batch = self._batch_to_gpu(batch, hparams.dataset_load_async)

            data_dict, lengths = batch
            batch_size = len(next(iter(lengths.values())))
            model.init_hidden(batch_size)

            # Compute max length because DataParallel splits the seq_lengths_input and padding will be done according to
            # the maximum length of that subset. Combining multi GPU output will fail with a size miss match.
            # https://pytorch.org/docs/stable/notes/faq.html#pack-rnn-unpack-with-data-parallelism
            max_lengths = dict()
            for key in data_dict.keys():
                if key in lengths:
                    l_max = max(lengths[key])
                    if hparams.use_gpu and hparams.num_gpus > 1:
                        l_max = l_max.repeat(hparams.num_gpus)
                    max_lengths[key] = l_max

            # Give max length because DataParallel splits the seq_lengths_input and padding will be done according to
            # the maximum length of that subset. Combining multi GPU output will fail with a size miss match.
            # https://pytorch.org/docs/stable/notes/faq.html#pack-rnn-unpack-with-data-parallelism
            if training:
                model(data_dict, lengths, max_lengths)
            else:
                with torch.no_grad():
                    model(data_dict, lengths, max_lengths)

            losses = {}
            for loss_fn in self.losses:
                loss_ = loss_fn(data_dict, lengths, total_steps)
                for loss_name, l in loss_.items():
                    if torch.isnan(l):
                        raise ValueError("Found NaN in {} loss.".format(loss_name))
                    if not hparams.replace_inf_grads_by_zero and torch.isinf(l):
                        raise ValueError("Found +/-Inf in {} loss.".format(loss_name))
                    if loss_name in losses:
                        raise KeyError("Loss with name {} defined twice.".format(loss_name))
                    losses[loss_name] = l
            backprop_loss = self.get_summed_losses_subset(
                loss_names=hparams.backprop_loss_names, losses=losses)
            if hparams.backprop_loss_names is None \
                    and hparams.scheduler_loss_names is None:
                scheduler_loss = backprop_loss.detach()
            else:
                scheduler_loss = self.get_summed_losses_subset(
                    loss_names=hparams.scheduler_loss_names, losses=losses).detach()

            if training:
                self.optimiser.zero_grad()
                backprop_loss.backward(retain_graph=hparams.backward_retain_graph)
                total_steps += 1

                # for params in reversed(list(self.model.parameters())):
                #     nan_or_inf |= torch.isnan(params.grad).any()
                #     nan_or_inf |= (params.grad == float("inf")).any()
                #     nan_or_inf |= (params.grad == -float("inf")).any()
                #     if nan_or_inf:
                #         raise ValueError("Found NaN/Inf in {}.".format(params))
                #         pdb.set_trace()

                if hparams.replace_inf_grads_by_zero:
                    self._replace_inf_grads_by_zero()

                if hparams.grad_clip_norm_type is not None:
                    # Adds a small bias.
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                   hparams.grad_clip_max_norm,
                                                   hparams.grad_clip_norm_type)
                if hparams.grad_clip_thresh is not None:
                    # Adds a big bias.
                    torch.nn.utils.clip_grad_value_(self.model.parameters(),
                                                    hparams.grad_clip_thresh)

                self.optimiser.step()

                # Update exponential moving average.
                if self.ema:
                    self.ema.update_params(model)

                current_iter = self._get_current_iteration(
                    batch_index=batch_index, current_epoch=current_epoch,
                    dataloader_length=len(dataloader), hparams=hparams,
                    total_epoch=total_epoch)
                self.run_scheduler(hparams=hparams, loss=scheduler_loss,
                                   current_iter=current_iter)

            # Logging current error.
            if batch_index % logging_batch_index == 0:
                log_message = "Train " if training else "Test "
                log_message += "mini batch [{:{front_pad}d}/{}]".format(
                    batch_index + 1, len(dataloader),
                    front_pad=len(str(len(dataloader))))
                log_message += "\tLoss: "
                log_message += " ".join(["{}: {:.3f}".format(key, loss) for
                                         key, loss in losses.items()])
                if hparams.log_memory_consumption:
                    log_message += "\tCPU: {:.0f} MB, ".format(
                        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e3)
                    if hparams.use_gpu:
                        log_message += "GPU: {} MB".format(
                            str(get_gpu_memory_map()))

                self.logger.info(log_message)

            losses = {k: l.detach() for k, l in losses.items()}
            for key, loss in losses.items():
                if key not in total_losses:
                    total_losses[key] = loss
                else:
                    total_losses[key] += loss

            if tb_writer is not None:
                tb_writer.add_scalars("Train loss", losses, total_steps)

            del data_dict, lengths, max_lengths, losses, backprop_loss, scheduler_loss

        total_losses = {key: value / len(dataloader) for key, value in total_losses.items()}

        if not training:
            if tb_writer is not None:
                tb_writer.add_scalars("Validation loss", total_losses, total_steps)

            self.logger.info(
                'Validation set: Total loss: {}\nAverage loss:\n\t{}\n'.format(
                    sum(total_losses.values()),
                    "\n\t".join(["{}: {:.3f}".format(key, loss)
                                 for key, loss in total_losses.items()])))

            fn_log_per_test = getattr(self.model, "log_per_test", None)
            if callable(fn_log_per_test):
                fn_log_per_test()

        np_total_losses = {key: loss.cpu().numpy() for key, loss in total_losses.items()}
        del total_losses

        return np_total_losses

    def _batch_to_gpu(self, batch: Tuple[dict, dict], load_async: bool):
        self.logger.debug("Convert batch to GPU.")
        data_dict, length_dict = batch
        data_dict = {k: self._to_cuda(v, load_async) for k, v in data_dict.items()}
        length_dict = {k: self._to_cuda(v, load_async) for k, v in length_dict.items()}
        return data_dict, length_dict

    @staticmethod
    def _to_cuda(element: torch.Tensor, load_async: bool):
        if (element is not None and hasattr(element, "cuda") and callable(element.cuda)):
            return element.cuda(non_blocking=load_async)
        else:
            return element

    def _replace_inf_with_zeros(self):
        """Replace inf/-inf in gradients of current model by 0."""
        for params in self.model.parameters():
            indices = (params.grad == float("inf"))
            if indices.any():
                self.logger.warning(
                    "Replace inf by 0.0 in the gradient of " + params)
                params.grad[indices] = 0.0
            indices = (params.grad == -float("inf"))
            if indices.any():
                self.logger.warning(
                    "Replace -inf by 0.0 in the gradient of " + params)
                params.grad[indices] = 0.0

    @staticmethod
    def get_summed_losses_subset(
            loss_names: List[str],
            losses: Dict[str, Union[np.ndarray, torch.Tensor]]) -> torch.Tensor:
        """
        Returns the sum of the losses given in the loss_names list.
        If loss_names is None, all losses are summed.
        """

        if loss_names is None:
            loss = sum(losses.values())
        else:
            loss = sum([losses[name] for name in loss_names])
        return loss

    def run_scheduler(self,
                      hparams: ExtendedHParams,
                      loss: torch.Tensor,
                      current_iter: int = None,
                      current_epoch: int = None) -> None:

        if self.scheduler is None:
            return

        if hparams.iterations_per_scheduler_step:
            if current_iter is None \
                    or current_iter % hparams.iterations_per_scheduler_step != 0:
                return
        elif hparams.epochs_per_scheduler_step:
            if current_epoch is None \
                    or current_epoch % hparams.epochs_per_scheduler_step != 0:
                return
        else:
            raise ValueError("Scheduler {} is defined but neither "
                             "hparams.iteration_per_scheduler_step nor "
                             "hparams.epochs_per_scheduler_step is set.".format(
                                 hparams.scheduler_type))

        self.logger.debug("Call scheduler.")
        self._scheduler_step_fn(loss)

    @staticmethod
    def _get_current_iteration(batch_index: int, current_epoch: int,
                               dataloader_length: int, hparams: ExtendedHParams,
                               total_epoch: int):
        if hparams.use_saved_learning_rate:
            assert total_epoch is not None
            return (total_epoch - 1) * dataloader_length + batch_index + 1
        else:
            assert current_epoch is not None
            return (current_epoch - 1) * dataloader_length + batch_index + 1

    def inference(self, data: dict, hparams: ExtendedHParams, seq_lengths: dict):
        self.model.eval()

        data_torch = {k: self._to_torch(v) for k, v in data.items()}
        seq_lengths_torch = {k: self._to_torch(v) for k, v in seq_lengths.items()}
        max_lengths_torch = {k: max(v) for k, v in seq_lengths.items()}
        batch_size = len(next(iter(seq_lengths.values())))

        if hparams.use_gpu:
            data_torch, seq_lengths_torch = self._batch_to_gpu(
                (data_torch, seq_lengths_torch),
                load_async=hparams.dataset_load_async)

        self.model.init_hidden(batch_size)
        with torch.no_grad():
            self.model.inference(data_torch, seq_lengths_torch, max_lengths_torch)

        # Return only new outputs.
        data = {}
        for key, value in data_torch.items():
            if not key.startswith('_'):
                data[key] = ModularModelHandlerPyTorch._return_values_to_numpy(
                    value, hparams.use_gpu)
        seq_lengths = {}
        for key, value in seq_lengths_torch.items():
            if not key.startswith('_'):
                seq_lengths[key] = ModularModelHandlerPyTorch._return_values_to_numpy(
                    value, hparams.use_gpu)

        return data, seq_lengths

    @staticmethod
    def _to_torch(element):
        if isinstance(element, np.ndarray):
            return torch.from_numpy(element)
        else:
            return element

    @staticmethod
    def _return_values_to_numpy(return_values, from_gpu):
        """
        Convert all tensors in return_values to CPU.
        return_values can be a iterable collections.
        Object which are not tensors are just returned.
        """
        if return_values is None:
            return None

        try:
            return return_values.detach().cpu().numpy()
        except AttributeError:
            try:
                return tuple(map(lambda x: ModularModelHandlerPyTorch._return_values_to_numpy(x, from_gpu),
                                 return_values))
            except TypeError:
                return return_values
