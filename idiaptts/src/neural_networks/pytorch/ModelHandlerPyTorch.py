#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#

"""
Handler module to work with PyTorch models.
"""

# System imports.
import sys
import os
import resource
import importlib
import warnings
from datetime import datetime, timezone
from operator import itemgetter
import logging
import copy
import re

# Third-party imports.

# Local source tree imports.
from idiaptts.src.neural_networks.ModelHandler import ModelHandler


class ModelHandlerPyTorch(ModelHandler):
    pass
    # """
    # Provides functionality to work with multiple network architectures. It allows to create, load and save a model,
    # train and test it and load the data for it. This class creates a wrapper around the framework used to implement
    # the models. This class should be implemented for each framework, this is the class for PyTorch.
    # """
    # logger = logging.getLogger(__name__)
    # # logger = multiprocessing.log_to_stderr()

    # #########################
    # # Default constructor
    # #
    # # def __init__(self):

    # #     super().__init__()

    # #     self.model_type = None
    # #     self.dim_in = None
    # #     self.dim_out = None

    # #     self.optimiser = None
    # #     self.scheduler = None

    # #     self._scheduler_step_fn = None
    # #     self.ema = None  # Exponential moving average object.

    # # @staticmethod
    # # def cuda_is_available():
    # #     return torch.cuda.is_available()

    # # @staticmethod
    # # def device_count():
    # #     return torch.cuda.device_count()

    # # @staticmethod
    # # def seed(seed):
    # #     torch.manual_seed(seed)

    # @staticmethod
    # def unsorted_pad_sequence(sequence, batch_first, seq_lengths=None):
    #     if seq_lengths is None:
    #         seq_lengths = torch.tensor(list(map(len, sequence)), dtype=torch.long)

    #     # Check if longest input sequence is also longest output sequence.
    #     if torch.argmax(seq_lengths) != 0:
    #         _, sort_indices = torch.sort(seq_lengths, descending=True)
    #         sequence_sorted = itemgetter(*sort_indices)(sequence)  # itemgetter is faster than lambda function.
    #         sequence = pad_sequence(sequence_sorted, batch_first)
    #         unsort_indices = tuple(np.argsort(sort_indices.numpy()))  # argsort function not implemented in PyTorch.
    #         if batch_first:
    #             sequence = sequence[unsort_indices, :]
    #         else:
    #             sequence = sequence[:, unsort_indices]
    #     else:
    #         sequence = pad_sequence(sequence, batch_first)

    #     return sequence.contiguous()

    # @staticmethod
    # def prepare_batch(batch, common_divisor=1, batch_first=False):
    #     """
    #     Convert a list of (input, target) tuples to a sorted batch.

    #     :param batch:                 List of (input, target, *extra) tuples, where target can be None.
    #     :param common_divisor:        Batch is trimmed so that dividable by the common_divisor (usually number of GPUs).
    #     :param batch_first:           Use the first dimension as batch dimension.
    #     :return:                      Returns input, target, seq_length_input, seq_length_output, mask,
    #                                   where target, seq_length_ouput and mask are None when target in input tuple is None
    #                                   and mask is also None if batch length is 1.
    #     """

    #     # T = 17*200
    #     # batch = [(np.random.random((T, input.shape[-1])).astype(input.dtype), np.random.random((T, target.shape[-1])).astype(target.dtype), *rest) for input, target, *rest in batch]

    #     # Remove samples if not equally dividable by given divisor (usually number of GPUs).
    #     # Remove before sorting to keep it unbiased.
    #     assert(len(batch) >= common_divisor)
    #     remainder = len(batch) % common_divisor
    #     if remainder > 0:
    #         batch = batch[:-remainder]

    #     # Sort batch if it contains more than one sample (longest first).
    #     permutation = None
    #     if len(batch) > 1:
    #         batch_enumerated = [(batch[i], i) for i in range(len(batch))]
    #         batch_enumerated.sort(key=lambda x: len(x[0][0]), reverse=True)
    #         batch, permutation = zip(*batch_enumerated)

    #     # Read lengths from samples.
    #     seq_lengths_input = torch.tensor([len(x[0]) for x in batch], dtype=torch.long)

    #     # V1: This version computes unnecessary padded zeros in forward pass of non-recurrent layers.
    #     inputs = [torch.from_numpy(x[0]) for x in batch]
    #     inputs = pad_sequence(inputs, batch_first)

    #     targets = None
    #     seq_lengths_target = None
    #     mask = None
    #     # Check if target is given.
    #     if batch[0][1] is None:
    #         # If no target is given, output lengths are assumed to be the same as input lengths.
    #         # If this is not the case, consider writing your own batch_collate_fn in your trainer.
    #         seq_lengths_target = seq_lengths_input
    #     else:
    #         # Read lengths from samples.
    #         seq_lengths_target = torch.tensor([len(x[1]) for x in batch], dtype=torch.long)

    #         # V1: This version computes unnecessary padded zeros in forward pass of non-recurrent layers.
    #         targets = [torch.from_numpy(x[1]) for x in batch]
    #         targets = ModelHandlerPyTorch.unsorted_pad_sequence(targets, batch_first, seq_lengths_target)

    #     # Crate a mask for the loss, target shape will be (T x B x C) or (B x T x C).
    #     # A mask can only be created when targets are given, but it should also be only necessary when
    #     # computing a loss which requires targets anyway.
    #     if len(batch) > 1 and targets is not None:
    #         mask = ModelHandlerPyTorch.sequence_mask(seq_lengths_target, seq_lengths_target.max(), batch_first=batch_first)
    #         # mask = mask.unsqueeze(-1).expand_as(targets).contiguous()
    #         # Ignore the mask if all entries are 1.
    #         if mask.min() == 1:
    #             mask = None

    #     extra_labels = list()
    #     for labels in list(zip(*map(lambda x: map(torch.from_numpy, x[2:]), batch))):
    #         if len(labels) > 1:  # Only pad when there is more than one in the batch.
    #             try:
    #                 extra_labels.append(ModelHandlerPyTorch.unsorted_pad_sequence(labels, batch_first))
    #             except RuntimeError as e:
    #                 extra_labels.append(labels)  # When extra_labels cannot be padded just add them.
    #         else:
    #             extra_labels.append(labels)

    #     return (inputs, targets, seq_lengths_input, seq_lengths_target, mask, permutation, *extra_labels)

    #     # # V2: Concat batch to one vector. Requires handling in each model with recurrence.
    #     # time_dim = 1 if batch_first else 0
    #     # if batch_first:
    #     #     batch = list(map(lambda x: (x[0][None, ...], x[1][None, ...]), batch))  # Add a batch dimension.
    #     # else:
    #     #     batch = list(map(lambda x: (x[0][:, None, ...], x[1][:, None, ...]), batch))  # Add a batch dimension.
    #     # inputs = torch.from_numpy(np.concatenate([x[0] for x in batch], axis=time_dim))
    #     # targets = torch.from_numpy(np.concatenate([x[1] for x in batch], axis=time))

    #     # # V3: The following will work when all layers accept PackedSequence as input.
    #     # return pack_sequence([torch.from_numpy(x[0]) for x in batch]),
    #     #        pack_sequence([torch.from_numpy(x[1]) for x in batch])

    # @staticmethod
    # def sequence_mask(sequence_length, max_len=None, batch_first=False):
    #     """Code adapted from https://github.com/r9y9/wavenet_vocoder/blob/master/train.py."""
    #     # Get some dimensions and sizes.
    #     time_dim = 1 if batch_first else 0
    #     if max_len is None:
    #         max_len = sequence_length.data.max()
    #     batch_size = len(sequence_length)

    #     # Create a range from 0 to max.
    #     seq_range = torch.arange(0, max_len, dtype=torch.long)
    #     # Expand the range to all samples in the batch.
    #     if batch_first:
    #         seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    #     else:
    #         seq_range_expand = seq_range.unsqueeze(1).expand(max_len, batch_size)

    #     # Expand the lengths of all samples to the full size.
    #     # seq_length_expand = torch.tensor(sequence_length, dtype=seq_range_expand.dtype, device=seq_range_expand.device).unsqueeze(time_dim).expand_as(seq_range_expand)
    #     seq_length_expand = sequence_length.detach().clone()\
    #         .type(seq_range_expand.dtype).to(device=seq_range_expand.device)\
    #         .detach().unsqueeze(time_dim).expand_as(seq_range_expand)

    #     # Compare element by element and return it as float.
    #     return (seq_range_expand < seq_length_expand).unsqueeze(-1).contiguous().float()

    # def set_dataset(self, hparams, dataset_train, dataset_val, collate_fn=None):
    #     common_divisor = hparams.num_gpus  # Will be 1 if used on CPU.
    #     collate_fn = self.prepare_batch if collate_fn is None else collate_fn
    #     num_workers = hparams.dataset_num_workers_gpu if hparams.use_gpu else hparams.dataset_num_workers_cpu

    #     self.dataloader_train = DataLoader(dataset=dataset_train,
    #                                        batch_size=hparams.batch_size_train,
    #                                        shuffle=hparams.shuffle_train_set,
    #                                        num_workers=num_workers,
    #                                        collate_fn=partial(collate_fn,
    #                                                           common_divisor=common_divisor,
    #                                                           batch_first=hparams.batch_first),
    #                                        pin_memory=hparams.dataset_pin_memory)
    #     self.dataloader_val = DataLoader(dataset_val,
    #                                      batch_size=hparams.batch_size_val,  # Used to be batch_size_test, please change it in your My* class.
    #                                      shuffle=hparams.shuffle_val_set,
    #                                      num_workers=num_workers,
    #                                      collate_fn=partial(collate_fn,
    #                                                         common_divisor=common_divisor,
    #                                                         batch_first=hparams.batch_first),
    #                                      pin_memory=hparams.dataset_pin_memory)

    # def set_optimiser(self, hparams, reset=False):
    #     """Initialise a PyTorch optimiser here."""
    #     if self.optimiser is None or reset:
    #         if hparams.optimiser is None:
    #             self.logger.info("Create {} optimiser.".format(hparams.optimiser_type))

    #             if "lr" not in hparams.optimiser_args:  # Backwards compatible.
    #                 try:
    #                     hparams.optimiser_args["lr"] = hparams.learning_rate
    #                 except AttributeError:
    #                     raise AttributeError("Learning rate not defined in hparams.optimiser_args[\"lr\"]")
    #             # Model is new.
    #             if hparams.optimiser_type == "Adam":
    #                 self.optimiser = torch.optim.Adam(self.model.parameters(), **hparams.optimiser_args)
    #             elif hparams.optimiser_type == "SGD":
    #                 self.optimiser = torch.optim.SGD(self.model.parameters(), **hparams.optimiser_args)
    #             # TODO: Implement the others here.
    #             else:
    #                 raise NotImplementedError("Optimiser type {} is not implemented.".format(hparams.optimiser_type))
    #         else:
    #             self.optimiser = hparams.optimiser(self.model.parameters())

    #     # Model was loaded from checkpoint, override learning rate if requested.
    #     if not hparams.use_saved_learning_rate and "lr" in hparams.optimiser_args:
    #         for g in self.optimiser.param_groups:
    #             g['lr'] = hparams.optimiser_args["lr"]

    # def set_scheduler(self, hparams, current_epoch=None, current_step=None, reset=False):
    #     """Initialise a PyTorch scheduler here."""
    #     if self.scheduler is not None and not reset:
    #         return
    #     if hparams.scheduler is None:
    #         if hparams.scheduler_type.lower() == "none":
    #             return
    #         assert hparams.scheduler_type != "default", "Please define a default scheduler type in the trainer class."

    #         self.logger.info("Create {} scheduler.".format(hparams.scheduler_type))

    #         # PyTorch schedulers use -1 as first epoch and call step immediately.
    #         if current_epoch == 0:
    #             current_epoch = -1
    #         if current_step == 0:
    #             current_step = -1

    #         if hparams.scheduler_type == "Plateau":
    #             self.scheduler = ReduceLROnPlateau(self.optimiser, **hparams.scheduler_args)
    #             self._scheduler_step_fn = self._scheduler_step_with_loss
    #             if hparams.epochs_per_scheduler_step is None and hparams.iterations_per_scheduler_step is None:
    #                 hparams.epochs_per_scheduler_step = 1
    #             return

    #         if current_step is None and self.dataloader_train is not None:
    #             current_step = max((current_epoch - 1) * len(self.dataloader_train), -1)
    #         if hparams.scheduler_type == "ExtendedExponentialLR":

    #             if hparams.epochs_per_scheduler_step is None:
    #                 if hparams.iterations_per_scheduler_step is None:
    #                     hparams.iterations_per_scheduler_step = 1
    #                 self.scheduler = ExtendedExponentialLR(self.optimiser, last_epoch=current_step,
    #                                                        **hparams.scheduler_args)
    #             else:
    #                 self.scheduler = ExtendedExponentialLR(self.optimiser, last_epoch=current_epoch - 1,
    #                                                        **hparams.scheduler_args)
    #             self._scheduler_step_fn = self._scheduler_step
    #             return

    #         if hparams.scheduler_type == "Exponential":
    #             if hparams.epochs_per_scheduler_step is None:
    #                 if hparams.iterations_per_scheduler_step is None:
    #                     hparams.iterations_per_scheduler_step = 1
    #                 self.scheduler = ExponentialLR(self.optimiser, last_epoch=current_step,
    #                                                **hparams.scheduler_args)
    #             else:
    #                 self.scheduler = ExponentialLR(self.optimiser, last_epoch=current_epoch - 1,
    #                                                **hparams.scheduler_args)
    #             self._scheduler_step_fn = self._scheduler_step
    #             return

    #         elif hparams.scheduler_type == "Noam":
    #             assert "wormup_steps" in hparams.scheduler_args, "Please define wormup_steps in hparams.scheduler_args."

    #             def noam_decay(iteration):
    #                 wormup_steps = float(hparams.scheduler_args['wormup_steps'])
    #                 return wormup_steps**0.5 * np.minimum((iteration + 1) * wormup_steps**-1.5,
    #                                                       (iteration + 1)**-0.5)

    #             if hparams.epochs_per_scheduler_step is None:
    #                 if hparams.iterations_per_scheduler_step is None:
    #                     hparams.iterations_per_scheduler_step = 1
    #                 self.scheduler = LambdaLR(self.optimiser, noam_decay, last_epoch=current_step)
    #             else:
    #                 self.scheduler = LambdaLR(self.optimiser, noam_decay, last_epoch=current_epoch - 1)
    #             self._scheduler_step_fn = self._scheduler_step
    #             return

    #         # TODO: Implement the others here.
    #         raise NotImplementedError("Scheduler type {} is not implemented.".format(hparams.scheduler_type))
    #     else:
    #         self.scheduler = hparams.scheduler(self.optimiser)

    # def _scheduler_step_with_loss(self, loss):
    #     self.scheduler.step(loss)

    # def _scheduler_step(self, loss):
    #     self.scheduler.step()

    # def create_model(self, hparams, dim_in, dim_out):

    #     # Use model factory to create the model.
    #     self.logger.info("Create network of type: " + hparams.model_type)
    #     self.model = ModelFactory.create(hparams.model_type, dim_in, dim_out, hparams)
    #     self.model_type = hparams.model_type
    #     self.dim_in = dim_in
    #     self.dim_out = dim_out

    #     if hparams.model_name is not None:
    #         self.model_name = hparams.model_name

    # @staticmethod
    # def save_model(file_path, model, model_type, dim_in, dim_out, verbose=False):
    #     """Saves all information needed to recreate the same model."""
    #     torch.save({'model_type': model_type,
    #                 'dim_in': dim_in,
    #                 'dim_out': dim_out,
    #                 'model_state_dict': model.state_dict()},
    #                file_path)
    #     if verbose:
    #         logging.info("Save model to {}.".format(file_path))

    # @staticmethod
    # def save_full_model(file_path, model, verbose=False):
    #     torch.save({'model': model}, file_path)
    #     if verbose:
    #         logging.info("Save full model to {}.".format(file_path))

    # @staticmethod
    # def load_model(file_path, hparams, ignore_layers=True, verbose=True):
    #     assert file_path is not None, "Path to model is None."

    #     created_model_type = None
    #     dim_in = None
    #     dim_out = None

    #     checkpoint = torch.load(file_path, map_location=lambda storage, loc: storage)
    #     try:
    #         model, created_model_type, (dim_in, dim_out) = ModelHandlerPyTorch._create_model(checkpoint, hparams)

    #         if verbose:
    #             mod_time = local_modification_time(file_path)
    #             logging.info("Load model state dict from {} (last modified {})"
    #                          .format(file_path, mod_time)
    #                          + (" ignoring {}.".format(hparams.ignore_layers) if ignore_layers
    #                                                                              and len(hparams.ignore_layers) > 0
    #                                                                           else "."))
    #         ModelHandlerPyTorch._load_model_state_dict(checkpoint, model, hparams, ignore_layers=ignore_layers)

    #     except KeyError:  # Ensure backwards compatibility.
    #         model = checkpoint['model']
    #         if ignore_layers and hasattr(hparams, "ignore_layers") and len(hparams.ignore_layers) > 0:
    #             logging.warning("Model was loaded as a whole. Cannot ignore {}".format(hparams.ignore_layers))

    #     if hparams.use_gpu:
    #         if hasattr(model, "set_gpu_flag") and callable(model.set_gpu_flag):
    #             model.set_gpu_flag(hparams.use_gpu)
    #         model = model.cuda()

    #     return model, created_model_type, (dim_in, dim_out)

    # @staticmethod
    # def _create_model(checkpoint, hparams, verbose=True):

    #     created_model_type = ModelHandlerPyTorch._get_valid_model_type(checkpoint, hparams)
    #     dim_in = checkpoint['dim_in']
    #     dim_out = checkpoint['dim_out']

    #     model = ModelFactory.create(created_model_type, dim_in, dim_out, hparams, verbose)

    #     return model, created_model_type, (dim_in, dim_out)

    # @staticmethod
    # def _get_valid_model_type(checkpoint, hparams):
    #     expected_model_type = hparams.model_type
    #     loaded_model_type = checkpoint['model_type']

    #     if expected_model_type is None:
    #         return loaded_model_type
    #     elif expected_model_type != loaded_model_type:
    #         warnings.warn("Expected type in hparams ({}) and loaded type ({}) do not match."
    #                       .format(expected_model_type, loaded_model_type))
    #         # raise TypeError("Expected type in hparams ({}) and loaded type ({}) should match."
    #         #                 .format(expected_model_type, loaded_model_type))

    #     return expected_model_type

    # @staticmethod
    # def _load_model_state_dict(checkpoint, model, hparams, ignore_layers=True):

    #     loaded_model_dict = checkpoint['model_state_dict']
    #     if ignore_layers:
    #         loaded_model_dict = ModelHandlerPyTorch._remove_ignored_layers(loaded_model_dict, model, hparams)
    #     model.load_state_dict(loaded_model_dict)

    # @staticmethod
    # def _remove_ignored_layers(model_dict, model, hparams):
    #     keys_to_pop = []
    #     if hasattr(hparams, "ignore_layers") and len(hparams.ignore_layers) > 0:
    #         for ignored_layer in hparams.ignore_layers:
    #             found_key = False
    #             for key in model_dict.keys():
    #                 if re.match(ignored_layer, key):
    #                     found_key = True
    #                     keys_to_pop.append(key)
    #             if not found_key:
    #                 raise KeyError("Cannot find layer {} in saved model dict: {}"
    #                                .format(ignored_layer, ", ".join(model_dict.keys())))

    #         model_dict = {k: v for k, v in model_dict.items() if k not in keys_to_pop}
    #         org_dict = model.state_dict()
    #         org_dict.update(model_dict)
    #         model_dict = org_dict
    #     return model_dict

    # def load_checkpoint(self, model_path, hparams, ignore_layers=True, load_optimiser=True, initial_lr=None):
    #     """
    #     Load a checkpoint, also transfers model and optimiser to GPU if hparams.use_gpu is True.

    #     :param file_path:         Full path to checkpoint.
    #     :param hparams:           Hyper-parameter container. Has to contain use_gpu key.
    #     :param ignore_layers:     Flag to determine if layers specified in hparams.ignore_layers should be ignored.
    #     :param initial_lr:        Initial learning rate of the model. Required by some schedulers to compute the
    #                               learning rate of the current epoch/iteration.
    #     :return:                  The number of epochs the loaded model was trained already.
    #     """
    #     self.logger.info("Load checkpoint from {}.".format(model_path))

    #     # Load model from checkpoint (and to GPU).
    #     self.model, self.model_type, (self.dim_in, self.dim_out) = ModelHandlerPyTorch.load_model(model_path,
    #                                                                                               hparams,
    #                                                                                               ignore_layers,
    #                                                                                               verbose=True)

    #     if hparams.ema_decay:
    #         self._load_ema_model(model_path, hparams)

    #     # The lambda expression makes it irrelevant if the checkpoint was saved from CPU or GPU.
    #     checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)

    #     # Load remaining checkpoint information.
    #     self.model_name = checkpoint['model_name']

    #     if load_optimiser:
    #         self._load_optimiser(checkpoint, hparams)

    #     if hparams.use_gpu and self.optimiser is not None:
    #         self._optimiser_to_gpu()

    #     return checkpoint['epoch']

    # def _load_ema_model(self, file_path, hparams):
    #     try:
    #         average_model, *_ = ModelHandlerPyTorch.load_model(file_path + "_ema", hparams, verbose=True)
    #         self.ema = ExponentialMovingAverage(average_model, hparams.ema_decay)
    #     except FileNotFoundError:
    #         self.logger.warning("EMA is enabled but no EMA model can be found at {}. ".format(file_path + "_ema") +
    #                             "A new one will be created for training.")
    #         self.ema = None

    # def _load_optimiser(self, checkpoint, hparams):
    #     try:
    #         self.optimiser = checkpoint['optimiser']
    #         self.logger.warning("Loaded a fully saved optimiser instead of its state dict", DeprecationWarning)
    #     except KeyError:
    #         # Load the state_dict of the optimiser. This cannot ignore layers because they are stored with ids and not
    #         # names. If you ignore some layers and have changed their dimension (e.g. embeddings) you have to created a
    #         # new optimiser (see https://discuss.pytorch.org/t/load-optimizer-for-partial-parameters/2617).
    #         optimiser_state_dict = checkpoint['optimiser_state_dict']
    #         if optimiser_state_dict is not None:
    #             self.optimiser = None  # Reset current optimiser which is linked to old weights.
    #             self.set_optimiser(hparams)
    #             try:
    #                 self.optimiser.load_state_dict(optimiser_state_dict)
    #             except ValueError as e:
    #                 self.logger.warning("State dict for optimiser {} miss matches checkpoint's optimiser "
    #                                     "state dict: {}\nContinuing without loading optimiser instead."
    #                                     .format(hparams.optimiser_type, e))

    #     if self.optimiser is not None:
    #         self._update_optimiser_initial_lr(hparams.optimiser_args["lr"] if "lr" in hparams.optimiser_args
    #                                           else hparams.learning_rate)

    # def _update_optimiser_initial_lr(self, initial_lr):
    #     # Initial learning rate is required by some optimisers to compute the learning rate of the current epoch.
    #     if initial_lr is not None:
    #         for group in self.optimiser.param_groups:
    #             # if hasattr(group, 'initial_lr'):
    #             group.setdefault('initial_lr', initial_lr)

    # def _optimiser_to_gpu(self):
    #     # TODO: Is this still necessary in PyTorch > 1.0
    #     # self.optimiser.cuda()  # Not implemented in master, but here:
    #     # https://github.com/andreh7/pytorch/blob/235ce5ba688a49f804422226ddc62a721bb811e0/torch/optim/optimizer.py
    #     # Requires the following function.
    #     def _transform_state(optimiser, transformation, filter_func, state=None):
    #         """Applies ``transformation`` recursively to each item in ``self.state`` for which ``filter_func`` returns True.
    #         Arguments:
    #             transformation (item -> item): function to be applied to each item passing ``filter_func``.
    #             filter_func (item -> `bool`): function which must return True for each item to which ``transformation`` should be applied.
    #         """
    #         if state is None:
    #             state = optimiser.state
    #         for key, value in state.items():
    #             if isinstance(value, dict):
    #                 _transform_state(optimiser, transformation, filter_func, value)
    #             else:
    #                 if filter_func(value):
    #                     state[key] = transformation(value)

    #     _transform_state(self.optimiser, lambda t: t.cuda(), lambda t: torch.is_tensor(t))  # Doing it manually.

    # def save_checkpoint(self, file_path, epoch):
    #     """
    #     Save checkpoint which consists of epoch number, model type, in/out dimensions, model state dict, whole
    #     optimiser, etc. Also save the EMA separately when one exists.
    #     """
    #     assert file_path is not None, "Given file_path is None."
    #     self.logger.info("Save checkpoint to " + file_path)

    #     makedirs_safe(os.path.dirname(file_path))  # Create directory if necessary.

    #     checkpoint_dict = {'epoch': epoch,
    #                        'model_name': self.model_name,
    #                        'optimiser_state_dict': self.optimiser.state_dict() if self.optimiser is not None else None,
    #                        # 'loss_function': loss_function
    #                        }
    #     if self.model_type:
    #         checkpoint_dict.update({'model_type': self.model_type,
    #                                 'dim_in': self.dim_in,
    #                                 'dim_out': self.dim_out,
    #                                 'model_state_dict': self.model.state_dict()})
    #     else:
    #         checkpoint_dict.update({'model': self.model})  # Special case where model_type is not given.

    #     torch.save(checkpoint_dict, file_path)
    #     # TODO: Also save the random generator states:
    #     #       torch.random.get_rng_state()
    #     #       torch.random.set_rng_state()
    #     #       Same for random package?
    #     # TODO: Save scheduler_type in checkpoint as well.

    #     if self.ema:
    #         self.save_model(file_path + "_ema", self.ema.model, self.model_type, self.dim_in, self.dim_out, verbose=True)

    # def forward(self, in_tensor, hparams, batch_seq_lengths=None, target=None, extra_labels=[]):
    #     """Forward one example through the model.

    #     :param in_tensor:           PyTorch tensor or numpy array.
    #     :param hparams:             Hyper-parameter container.
    #     :param batch_seq_lengths:   Tuple with the real length (not padded) of each tensor in in_tensor.
    #     :param target:              PyTorch tensor or numpy array, used for teacher forcing, can be None.
    #     :return:                    Output of model (numpy array).
    #     """
    #     self.model.eval()

    #     # If input tensor is numpy array convert it to torch tensor.
    #     if isinstance(in_tensor, np.ndarray):
    #         in_tensor = torch.from_numpy(in_tensor)
    #     if hparams.use_gpu and in_tensor is not None:
    #         in_tensor = in_tensor.cuda()

    #     # If target tensor is given and numpy array, convert it to torch tensor.
    #     if target is not None:
    #         if isinstance(target, np.ndarray):
    #             target = torch.from_numpy(target)
    #         if hparams.use_gpu:
    #             target = target.cuda()

    #     # If lengths is tuple or numpy array convert it to torch tensor.
    #     if batch_seq_lengths is not None:
    #         if isinstance(batch_seq_lengths, tuple):
    #             batch_seq_lengths = torch.tensor(batch_seq_lengths, dtype=torch.long)
    #         if isinstance(batch_seq_lengths, np.ndarray):
    #             batch_seq_lengths = torch.from_numpy(batch_seq_lengths, dtype=torch.long)
    #     else:
    #         batch_seq_lengths = torch.tensor([len(in_tensor)], dtype=torch.long)

    #     output_extra_labels = list()
    #     for labels in (list(zip(*extra_labels)) if extra_labels is not None and len(extra_labels) > 1 else extra_labels):
    #         # TODO: A try-except might be needed here.
    #         if isinstance(labels, np.ndarray):
    #             labels = torch.from_numpy(labels)
    #         if hparams.use_gpu:
    #             labels = labels.cuda()
    #         output_extra_labels.append(labels)

    #     hidden = self.model.init_hidden(len(batch_seq_lengths))
    #     output, hidden = self.model(in_tensor, hidden, batch_seq_lengths, batch_seq_lengths[0], target, None, *output_extra_labels)

    #     # Convert back (to cpu and) to numpy.
    #     return ModelHandlerPyTorch._return_values_to_numpy(output, hparams.use_gpu),\
    #            ModelHandlerPyTorch._return_values_to_numpy(hidden, hparams.use_gpu)

    # @staticmethod
    # def _return_values_to_numpy(return_values, from_gpu):
    #     """
    #     Convert all tensors in return_values to CPU.
    #     return_values can be a iterable collections.
    #     Object which are not tensors are just returned.
    #     """
    #     if return_values is None:
    #         return None

    #     try:
    #         return return_values.detach().cpu().numpy()
    #     except AttributeError:
    #         try:
    #             return tuple(map(lambda x: ModelHandlerPyTorch._return_values_to_numpy(x, from_gpu), return_values))
    #         except TypeError:
    #             return return_values

    # def process_dataloader(self, dataloader, loss_function, hparams, total_epoch, current_epoch=None, training=True):
    #     """
    #     Train or test the model by loading batches from the dataloader.

    #     :param dataloader:        Dataloader of the train/test set.
    #     :param loss_function:     PyTorch function/class to compute loss.
    #     :param hparams:           Hyper-parameter container.
    #     :param total_epoch:       Total number of training epochs. Used to compute the iteration for some schedulers
    #                               when the learning rate is not reset in beginning of current train loop.
    #     :param current_epoch:     Number of epoch in current training loop. Used to compute the iteration for some
    #                               schedulers when the learning rate was reset in beginning of current train loop
    #     :param training:          Determines if it runs the training or testing loop.
    #     :return:                  Tuple of total loss and total loss per output feature.
    #     """

    #     model = self.model
    #     if training:
    #         model.train()
    #         self.logger.info("{}: Train with {} on {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    #                                                           self.optimiser,
    #                                                           str(torch.cuda.device_count()) + " GPU(s)." if hparams.use_gpu else "1 CPU."))
    #     else:
    #         if self.ema is not None:
    #             self.logger.info("Using averaged model for validation.")
    #             model = self.ema.model
    #         model.eval()
    #         self.logger.info("{}: Compute loss of validation set.".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    #     if hparams.log_memory_consumption:
    #         self.logger.info('CPU: {:.0f} MB, GPU: {} MB'
    #                          .format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e3,
    #                                  str(get_gpu_memory_map()) if hparams.use_gpu else "-"))

    #     # Multi-GPU support.
    #     if hparams.num_gpus > 1:
    #         model = DataParallel(model, dim=0 if hparams.batch_first else 1)
    #         model.init_hidden = model.module.init_hidden  # Make the init_hidden method directly accessible.

    #     # Log loss after each <hparams.logging_batch_index_perc>% of batches.
    #     logging_batch_index = (len(dataloader) // hparams.logging_batch_index_perc) + 1

    #     current_batch_index = -1  # Consider special case for first batch.
    #     current_batch = None
    #     loss = None
    #     total_loss = None
    #     loss_features = None

    #     # FIXME: Experimental implementation to pre-load the next batch to GPU. Does not work yet because it blocks.
    #     if hparams.use_gpu and hparams.preload_next_batch_to_gpu:
    #         # Create an iterator around the dataloader to pop the first element before the for loop.
    #         dataloader = iter(dataloader)
    #         # Pop the first batch from the dataloader.
    #         current_batch = next(dataloader)
    #         current_batch_index = 0
    #         # Move the first batch to GPU.
    #         current_batch = self._batch_to_gpu(current_batch, hparams.dataset_load_async)

    #     # Iterate on the batches.
    #     for next_batch_index, next_batch in enumerate(dataloader, current_batch_index + 1):

    #         # Move the next batch to GPU.
    #         if hparams.use_gpu:
    #             next_batch = self._batch_to_gpu(next_batch, hparams.dataset_load_async)

    #         # If there is no current batch either experiment is on CPU or hparams.preload_next_batch_to_gpu is False.
    #         # In any case use the "next" batch for the current iteration.
    #         if current_batch is None:
    #             current_batch_index = next_batch_index
    #             current_batch = next_batch

    #         # Get data and move it to gpu if necessary.
    #         inputs, target, seq_lengths_input, seq_lengths_target, mask, permutation, *extra_labels = current_batch

    #         # self.logger.info(str(torch.max(seq_lengths_input)) + " " + str(torch.max(seq_lengths_target)))

    #         # Request the architecture to initialise its hidden states.
    #         hidden = model.init_hidden(len(seq_lengths_input))

    #         # Forward the input through the model.
    #         max_length_inputs = seq_lengths_input[0]
    #         if hparams.use_gpu and hparams.num_gpus > 1:
    #             assert(len(seq_lengths_input) % torch.cuda.device_count() == 0)  # Batch is not equally dividable into the given number of GPUs.
    #             max_length_inputs = max_length_inputs.repeat(hparams.num_gpus)

    #         # Give max length because DataParallel splits the seq_lengths_input and padding will be done according to
    #         # the maximum length of that subset. Combining multi GPU output will fail with a size miss match.
    #         # https://pytorch.org/docs/stable/notes/faq.html#pack-rnn-unpack-with-data-parallelism
    #         if training:
    #             output, hidden_out = model(inputs,
    #                                        hidden,
    #                                        seq_lengths_input,
    #                                        max_length_inputs,
    #                                        target,
    #                                        seq_lengths_target,
    #                                        *extra_labels)
    #         else:
    #             with torch.no_grad():
    #                 output, hidden_out = model(inputs,
    #                                            hidden,
    #                                            seq_lengths_input,
    #                                            max_length_inputs,
    #                                            target if hparams.teacher_forcing_in_test else None,
    #                                            seq_lengths_target,
    #                                            *extra_labels)

    #         # Compute loss of the output.
    #         loss_full = loss_function(output, target)
    #         if type(loss_full) not in (list, tuple):
    #             loss_full = [loss_full]
    #         sample_loss_features = [None] * len(loss_full)

    #         for loss_idx, loss in enumerate(loss_full):
    #             assert loss.nelement() > 1, "Don't reduce the loss, so that the mask can be applied. " \
    #                                         "Use reduction='none' in loss function."
    #             if mask is not None:
    #                 loss = loss * mask  # Don't do inplace multiplication because both tensors could be expanded.

    #             if hparams.loss_per_sample:
    #                 # Average the loss on each sample of the batch and then compute the mean, which means that
    #                 # each sample in the batch contributes equally to the loss independently from its length.

    #                 # Automatically determine the time dimension and sum over it.
    #                 time_dim = 0 if loss.shape[0] == seq_lengths_target.max() else 1
    #                 # Take mean over batch dimension.
    #                 sample_loss_features[loss_idx] = (loss.sum(dim=time_dim) / seq_lengths_target.unsqueeze(-1).float()).mean(0)
    #                 loss = sample_loss_features[loss_idx].mean()
    #             else:
    #                 # Default: Average the loss over all frames, then compute the mean of all loss channels.
    #                 sample_loss_features[loss_idx] = (loss.sum(dim=(0, 1)) / sum(seq_lengths_target).float())
    #                 loss = sample_loss_features[loss_idx].mean()

    #             if torch.isnan(loss):
    #                 self.logger.error("Loss is nan: {}".format(sample_loss_features))
    #                 break
    #             # nan_or_inf = torch.isnan(loss)
    #             # for params in self.model.parameters():
    #             #     nan_or_inf |= torch.isnan(params.data).any()
    #             #     nan_or_inf |= (params.data == float("inf")).any()
    #             #     nan_or_inf |= (params.data == -float("inf")).any()
    #             #     if nan_or_inf:
    #             #         pdb.set_trace()
    #             loss_full[loss_idx] = loss

    #         if training:
    #             # Zero all gradients in the optimiser.
    #             self.optimiser.zero_grad()

    #             # Propagate first error backwards.
    #             loss_full[0].backward(retain_graph=hparams.backward_retain_graph)

    #             # # DEBUG: Check for NaNs and Infs and start pdb debugger if some are found.
    #             # nan_or_inf = torch.isnan(loss)
    #             # for params in reversed(list(self.model.parameters())):
    #             #     nan_or_inf |= torch.isnan(params.grad).any()
    #             #     nan_or_inf |= (params.grad == float("inf")).any()
    #             #     nan_or_inf |= (params.grad == -float("inf")).any()
    #             #     if nan_or_inf:
    #             #         pdb.set_trace()

    #             if hparams.replace_inf_grads_by_zero:
    #                 # Replace inf/-inf in gradients by 0.
    #                 for params in self.model.parameters():
    #                     # Check for positive infinity.
    #                     indices = (params.grad == float("inf"))
    #                     if indices.any():
    #                         self.logger.warning("Replace inf by 0.0 in the gradient of " + params)
    #                         params.grad[indices] = 0.0
    #                     # Check for negative infinity.
    #                     indices = (params.grad == -float("inf"))
    #                     if indices.any():
    #                         self.logger.warning("Replace -inf by 0.0 in the gradient of " + params)
    #                     params.grad[indices] = 0.0

    #             # Clip gradients.
    #             if hparams.grad_clip_norm_type is not None:
    #                 torch.nn.utils.clip_grad_norm_(self.model.parameters(), hparams.grad_clip_max_norm, hparams.grad_clip_norm_type)  # Adds a small bias.
    #             if hparams.grad_clip_thresh is not None:
    #                 torch.nn.utils.clip_grad_value_(self.model.parameters(), hparams.grad_clip_thresh)  # Adds a big bias.

    #             # Change all model weights depending on their gradient.
    #             self.optimiser.step()

    #             # Update moving average.
    #             if self.ema:
    #                 self.ema.update_params(model)

    #             # Run the scheduler_type if one exists and should be called after some iterations.
    #             if self.scheduler:
    #                 if hparams.iterations_per_scheduler_step:
    #                     if hparams.use_saved_learning_rate:
    #                         current_iteration = (total_epoch - 1) * len(dataloader) + current_batch_index + 1
    #                     else:
    #                         current_iteration = (current_epoch - 1) * len(dataloader) + current_batch_index + 1
    #                     if current_iteration % hparams.iterations_per_scheduler_step == 0:
    #                         self._scheduler_step_fn(loss.detach(), current_iteration)
    #                         # self.logger.info(str(self.optimiser))

    #         # Logging current error.
    #         if current_batch_index % logging_batch_index == 0:
    #             self.logger.info('{} [{:{front_pad}d}/{}]\tLoss: {}'
    #                              .format("Trained " if training else "Tested ",
    #                                      current_batch_index + 1,
    #                                      len(dataloader),
    #                                      " ".join(["{:.3f}".format(loss) for loss in loss_full]),
    #                                      front_pad=len(str(len(dataloader))))
    #                              + ("\tCPU: {:.0f} MB, GPU: {} MB"
    #                                 .format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e3,
    #                                         str(get_gpu_memory_map()) if hparams.use_gpu else "-"))
    #                              if hparams.log_memory_consumption else "")

    #             if training:
    #                 fn_log_per_batch = getattr(self.model, "log_per_batch", None)
    #                 if callable(fn_log_per_batch):
    #                     fn_log_per_batch()

    #         loss = list(map(lambda l: l.detach(), loss_full))
    #         # Accumulate loss.
    #         if total_loss is None:
    #             total_loss = loss
    #         else:
    #             total_loss = [tl + l for tl, l in zip(total_loss, loss)]
    #         # total_loss += loss
    #         if loss_features is None:
    #             loss_features = list(map(lambda l: l.detach(), sample_loss_features))
    #         else:
    #             loss_features = [lf + l.detach() for lf, l in zip(loss_features, sample_loss_features)]
    #             del sample_loss_features

    #         del inputs, target, hidden, seq_lengths_input, max_length_inputs, seq_lengths_target, mask
    #         del output, hidden_out, loss_full

    #         if hparams.use_gpu and hparams.preload_next_batch_to_gpu:
    #             # Use the next_batch as current_batch in next iteration.
    #             current_batch = next_batch
    #             current_batch_index = next_batch_index
    #         else:
    #             # Reset the current_batch to None so that in next iteration current_batch = next_batch again.
    #             current_batch = None
    #             # current_batch_index = None  # This is actually unnecessary.

    #     loss_features = [l / len(dataloader) for l in loss_features]
    #     total_loss = [l / len(dataloader) for l in total_loss]
    #     if not training:
    #         self.logger.info('Test set: Average loss: {}, error_features:\n{})\n'
    #                          .format(" ".join(["{:.3f}".format(loss) for loss in total_loss]),
    #                                  "\n".join(["{}".format(loss.view(1, -1)) for loss in loss_features])))

    #         fn_log_per_test = getattr(self.model, "log_per_test", None)
    #         if callable(fn_log_per_test):
    #             fn_log_per_test()

    #     np_total_loss = [l.cpu().numpy() for l in total_loss]
    #     np_loss_features = [l.cpu().numpy() for l in loss_features]
    #     del total_loss, loss_features

    #     return np_total_loss, np_loss_features

    # def _batch_to_gpu(self, batch, load_async):
    #     # output = []
    #     # for element in batch:
    #     #     if element is not None and hasattr(element, "cuda") and callable(element.cuda):
    #     #         logging.warning(element.shape)
    #     #         output.append(element.cuda(async=load_async))
    #     # return output
    #     return [element.cuda(async=load_async) if element is not None
    #                                               and hasattr(element, "cuda")
    #                                               and callable(element.cuda)
    #             else element for element in batch]

    # def test(self, hparams, total_epoch, current_epoch, loss_function):
    #     if hparams.use_gpu:
    #         assert (hparams.num_gpus <= torch.cuda.device_count())  # Specified number of GPUs is incorrect.

    #     return self.process_dataloader(self.dataloader_val,
    #                                    loss_function,
    #                                    hparams,
    #                                    total_epoch,
    #                                    current_epoch,
    #                                    training=False)

    # def train(self, hparams, total_epoch, current_epoch, loss_function):
    #     if hparams.use_gpu:
    #         assert (hparams.num_gpus <= torch.cuda.device_count())  # Specified number of GPUs is incorrect.

    #     if hparams.ema_decay and not self.ema:
    #         average_model = ModelFactory.create(self.model_type, self.dim_in, self.dim_out, hparams)
    #         average_model.load_state_dict(self.model.state_dict())
    #         self.ema = ExponentialMovingAverage(average_model, hparams.ema_decay)

    #     return self.process_dataloader(self.dataloader_train,
    #                                    loss_function,
    #                                    hparams,
    #                                    total_epoch,
    #                                    current_epoch,
    #                                    training=True)[0]

    # def run_scheduler(self, loss):
    #     if self.scheduler is not None:
    #         # self.logger.info("Call scheduler.")
    #         self._scheduler_step_fn(loss)
