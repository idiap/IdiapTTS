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
from datetime import datetime
from operator import itemgetter
import logging
import copy

# Third-party imports.
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader
from torch.nn import DataParallel

# Local source tree imports.
from idiaptts.misc.utils import get_gpu_memory_map
from idiaptts.src.neural_networks.ModelHandler import ModelHandler
from idiaptts.src.neural_networks.pytorch.models.RNNDyn import *
from idiaptts.misc.utils import makedirs_safe
from idiaptts.src.neural_networks.pytorch.ExponentialMovingAverage import ExponentialMovingAverage
from idiaptts.src.neural_networks.pytorch.ModelFactory import ModelFactory


class ModelHandlerPyTorch(ModelHandler):
    """
    Provides functionality to work with multiple network architectures. It allows to create, load and save a model,
    train and test it and load the data for it. This class creates a wrapper around the framework used to implement
    the models. This class should be implemented for each framework, this is the class for PyTorch.
    """
    logger = logging.getLogger(__name__)
    # logger = multiprocessing.log_to_stderr()

    #########################
    # Default constructor
    #
    def __init__(self):

        super().__init__()

        self.model_type = None
        self.dim_in = None
        self.dim_out = None

        self._scheduler_step_fn = None
        self.ema = None  # Exponential moving average object.

    @staticmethod
    def cuda_is_available():
        return torch.cuda.is_available()

    @staticmethod
    def device_count():
        return torch.cuda.device_count()

    @staticmethod
    def seed(seed):
        torch.manual_seed(seed)

    @staticmethod
    def prepare_batch(batch, common_divisor=1, batch_first=False):
        """
        Convert a list of (input, target) tuples to a sorted batch.

        :param batch:                 List of (input, target) tuples, where target can be None.
        :param common_divisor:        Batch is trimmed so that dividable by the common_divisor (usually number of GPUs).
        :param batch_first:           Use the first dimension as batch dimension.
        :return:                      Returns input, target, seq_length_input, seq_length_output, mask,
                                      where target, seq_length_ouput and mask are None when target in input tuple is None
                                      and mask is also None if batch length is 1.
        """

        # Remove samples if not equally dividable by given divisor (usually number of GPUs).
        # Remove before sorting to keep it unbiased.
        assert(len(batch) >= common_divisor)
        remainder = len(batch) % common_divisor
        if remainder > 0:
            batch = batch[:-remainder]

        # Sort batch if it contains more than one sample (longest first).
        permutation = None
        if len(batch) > 1:
            batch_enumerated = [(batch[i], i) for i in range(len(batch))]
            batch_enumerated.sort(key=lambda x: len(x[0][0]), reverse=True)
            batch, permutation = zip(*batch_enumerated)

        # Read lengths from samples.
        seq_lengths_input = torch.tensor([len(x[0]) for x in batch], dtype=torch.long)

        # V1: This version computes unnecessary padded zeros in forward pass of non-recurrent layers.
        inputs = [torch.from_numpy(x[0]) for x in batch]
        inputs = pad_sequence(inputs, batch_first)

        targets = None
        seq_lengths_target = None
        mask = None
        # Check if target is given.
        if batch[0][1] is None:
            # If no target is given, output lengths are assumed to be the same as input lengths.
            # If this is not the case, consider writing your own batch_decollate_fn in your trainer.
            seq_lengths_target = seq_lengths_input
        else:
            # Read lengths from samples.
            seq_lengths_target = torch.tensor([len(x[1]) for x in batch], dtype=torch.long)

            # V1: This version computes unnecessary padded zeros in forward pass of non-recurrent layers.
            targets = [torch.from_numpy(x[1]) for x in batch]

            # Check if longest input sequence is also longest output sequence.
            if torch.argmax(seq_lengths_target) != 0:
                seq_lengths_target_sorted, sort_indices = torch.sort(seq_lengths_target, descending=True)
                targets_sorted = itemgetter(*sort_indices)(targets)  # itemgetter is faster than lambda function.
                targets = pad_sequence(targets_sorted, batch_first)
                unsort_indices = tuple(np.argsort(sort_indices.numpy()))  # arsort function not implemented in PyTorch.
                if batch_first:
                    targets = targets[unsort_indices]
                else:
                    targets = targets[:, unsort_indices]
                targets = targets.contiguous()
            else:
                targets = pad_sequence(targets, batch_first)

        # Crate a mask for the loss, target shape will be (T x B x C) or (B x T x C).
        # A mask can only be created when targets are given, but it should also be only necessary when
        # computing a loss which requires targets anyway.
        if len(batch) > 1:
            mask = ModelHandlerPyTorch.sequence_mask(seq_lengths_target, seq_lengths_target.max(), batch_first=batch_first)
            # mask = mask.unsqueeze(-1).expand_as(targets).contiguous()
            # Ignore the mask if all entries are 1.
            if mask.min() == 1:
                mask = None

        return inputs, targets, seq_lengths_input, seq_lengths_target, mask, permutation

        # # V2: Concat batch to one vector. Requires handling in each model with recurrence.
        # time_dim = 1 if batch_first else 0
        # if batch_first:
        #     batch = list(map(lambda x: (x[0][None, ...], x[1][None, ...]), batch))  # Add a batch dimension.
        # else:
        #     batch = list(map(lambda x: (x[0][:, None, ...], x[1][:, None, ...]), batch))  # Add a batch dimension.
        # inputs = torch.from_numpy(np.concatenate([x[0] for x in batch], axis=time_dim))
        # targets = torch.from_numpy(np.concatenate([x[1] for x in batch], axis=time))

        # # V3: The following will work when all layers accept PackedSequence as input.
        # return pack_sequence([torch.from_numpy(x[0]) for x in batch]),
        #        pack_sequence([torch.from_numpy(x[1]) for x in batch])

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
        seq_length_expand = torch.tensor(sequence_length, dtype=seq_range_expand.dtype, device=seq_range_expand.device).unsqueeze(time_dim).expand_as(seq_range_expand)

        # Compare element by element and return it as float.
        return (seq_range_expand < seq_length_expand).unsqueeze(-1).contiguous().float()

    def set_dataset(self, hparams, dataset_train, dataset_val, collate_fn=None):
        common_divisor = hparams.num_gpus  # Will be 1 if used on CPU.
        self.dataloader_train = DataLoader(dataset=dataset_train,
                                           batch_size=hparams.batch_size_train,
                                           shuffle=hparams.shuffle_train_set,
                                           num_workers=hparams.dataset_num_workers_gpu if hparams.use_gpu else hparams.dataset_num_workers_cpu,
                                           collate_fn=partial(self.prepare_batch if collate_fn is None else collate_fn,
                                                              common_divisor=common_divisor,
                                                              batch_first=hparams.batch_first),
                                           pin_memory=hparams.dataset_pin_memory)
        self.dataloader_val = DataLoader(dataset_val,
                                         batch_size=hparams.batch_size_val,  # Used to be batch_size_test, please change it in your My* class.
                                         shuffle=hparams.shuffle_val_set,
                                         num_workers=hparams.dataset_num_workers_gpu if hparams.use_gpu else hparams.dataset_num_workers_cpu,
                                         collate_fn=partial(self.prepare_batch if collate_fn is None else collate_fn,
                                                            common_divisor=common_divisor,
                                                            batch_first=hparams.batch_first),
                                         pin_memory=hparams.dataset_pin_memory)

    def set_optimiser(self, hparams):
        """Initialise a PyTorch optimiser here."""
        if self.optimiser is None:
            if hparams.optimiser is None:
                if "lr" not in hparams.optimiser_args:  # Backwards compatible.
                    hparams.optimiser_args["lr"] = hparams.learning_rate
                # Model is new.
                if hparams.optimiser_type == "Adam":
                    self.optimiser = torch.optim.Adam(self.model.parameters(), **hparams.optimiser_args)
                    return
                if hparams.optimiser_type == "SGD":
                    self.optimiser = torch.optim.SGD(self.model.parameters(), **hparams.optimiser_args)
                    return
                # TODO: Implement the others here.
                raise NotImplementedError("Optimiser type {} is not implemented.".format(hparams.optimiser_type))
            else:
                self.optimiser = hparams.optimiser(self.model.parameters())

        # Model was loaded from checkpoint, override learning rate if requested.
        if not hparams.use_saved_learning_rate and "lr" in hparams.optimiser_args:
            for g in self.optimiser.param_groups:
                g['lr'] = hparams.optimiser_args["lr"]

    def set_scheduler(self, hparams, current_epoch):
        """Initialise a PyTorch scheduler here."""
        if hparams.scheduler is None:
            if hparams.scheduler_type == "None":
                return
            assert hparams.scheduler_type != "default", "Please define a default scheduler type in the trainer class."

            if hparams.scheduler_type == "Plateau":
                self.scheduler = ReduceLROnPlateau(self.optimiser, **hparams.scheduler_args)
                self._scheduler_step_fn = self._scheduler_step_with_loss
                if hparams.epochs_per_scheduler_step is None and hparams.iterations_per_scheduler_step is None:
                    hparams.epochs_per_scheduler_step = 1
                return

            current_iteration = max((current_epoch - 1) * len(self.dataloader_train), -1)
            if hparams.scheduler_type == "Exponential":
                if hparams.epochs_per_scheduler_step is None:
                    if hparams.iterations_per_scheduler_step is None:
                        hparams.iterations_per_scheduler_step = 1
                    self.scheduler = ExponentialLR(self.optimiser, last_epoch=current_iteration, **hparams.scheduler_args)
                else:
                    self.scheduler = ExponentialLR(self.optimiser, last_epoch=current_epoch - 1, **hparams.scheduler_args)
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
                    self.scheduler = LambdaLR(self.optimiser, noam_decay, last_epoch=current_iteration)
                else:
                    self.scheduler = LambdaLR(self.optimiser, noam_decay, last_epoch=current_epoch - 1)
                self._scheduler_step_fn = self._scheduler_step
                return

            # TODO: Implement the others here.
            raise NotImplementedError("Scheduler type {} is not implemented.".format(hparams.scheduler_type))
        else:
            self.scheduler = hparams.scheduler(self.optimiser)

    def _scheduler_step_with_loss(self, loss, iteration):
        self.scheduler.step(loss, iteration + 1)
        # self.logger.info("Epoch: " + str(epoch + 1) + ", lr: " + str(self.scheduler_type.get_lr()))

    def _scheduler_step(self, loss, iteration=None):
        self.scheduler.step(iteration)

    def create_model(self, hparams, dim_in, dim_out):

        # Use model factory to create the model.
        self.logger.info("Create network of type: " + hparams.model_type)
        self.model = ModelFactory.create(hparams.model_type, dim_in, dim_out, hparams)
        self.model_type = hparams.model_type
        self.dim_in = dim_in
        self.dim_out = dim_out

        if hparams.model_name is not None:
            self.logger.info("Selected network name: " + hparams.model_name)
            self.model_name = hparams.model_name

    @staticmethod
    def save_model(file_path, model, model_type, dim_in, dim_out, verbose=False):
        """Saves all information needed to recreate the same model."""
        torch.save({'model_type': model_type,
                    'dim_in': dim_in,
                    'dim_out': dim_out,
                    'model_state_dict': model.state_dict()},
                   file_path)
        if verbose:
            logging.info("Save model to {}.".format(file_path))

    @staticmethod
    def save_full_model(file_path, model, verbose=False):
        torch.save({'model': model}, file_path)
        if verbose:
            logging.info("Save full model to {}.".format(file_path))

    @staticmethod
    def load_model(file_path, hparams, verbose=True):
        dim_in = None
        dim_out = None
        model_type = None

        checkpoint = torch.load(file_path, map_location=lambda storage, loc: storage)
        try:
            expected_model_type = hparams.model_type
            model_type = checkpoint['model_type']
            if expected_model_type and expected_model_type != model_type:  # None when model should be loaded by name.
                raise TypeError("Expected type in hparams ({}) and loaded type ({}) should match."
                                .format(expected_model_type, model_type))
            hparams.model_type = model_type  # Use the loaded model type during creation in factory.
            dim_in = checkpoint['dim_in']
            dim_out = checkpoint['dim_out']
            model = ModelFactory.create(model_type, dim_in, dim_out, hparams, verbose)
            hparams.model_type = expected_model_type  # Can still be None.
            if verbose:
                logging.info("Load model state dict from {}".format(file_path) +
                             (" ignoring {}.".format(hparams.ignore_layers) if len(hparams.ignore_layers) > 0 else "."))

            model_dict = checkpoint['model_state_dict']
            if len(hparams.ignore_layers) > 0:
                model_dict = {k: v for k, v in model_dict.items()
                              if k not in hparams.ignore_layers}
                org_dict = model.state_dict()
                org_dict.update(model_dict)
                model_dict = org_dict
            model.load_state_dict(model_dict)
        except KeyError:  # Ensure backwards compatibility.
            model = checkpoint['model']
            if len(hparams.ignore_layers) > 0:
                logging.warning("Model was loaded as a whole. Cannot ignore {}".format(hparams.ignore_layers))

        if hparams.use_gpu:
            if hasattr(model, "set_gpu_flag") and callable(model.set_gpu_flag):
                model.set_gpu_flag(hparams.use_gpu)

        return model, model_type, dim_in, dim_out

    def load_checkpoint(self, file_path, hparams, initial_lr=0.0):
        """
        Load a checkpoint, also transfers model and optimiser to GPU if hparams.use_gpu is True.

        :param file_path:         Full path to checkpoint.
        :param hparams:           Hyper-parameter container. Has to contain use_gpu key.
        :param initial_lr:        Initial learning rate of the model. Required by some schedulers to compute the
                                  learning rate of the current epoch/iteration.
        :return:                  The number of epochs the loaded model was trained already.
        """
        self.logger.info("Load checkpoint from {}.".format(file_path))

        # Load model from checkpoint (and to GPU).
        self.model, self.model_type, self.dim_in, self.dim_out = self.load_model(file_path, hparams, verbose=True)

        if hparams.ema_decay:
            try:
                average_model, *_ = self.load_model(file_path + "_ema",
                                                    hparams,
                                                    verbose=True)
                self.ema = ExponentialMovingAverage(average_model, hparams.ema_decay)
            except FileNotFoundError:
                self.logger.warning("EMA is enabled but no EMA model can be found at {}. ".format(file_path + "_ema") +
                                    "A new one will be created for training.")
                self.ema = None

        # The lambda expression makes it irrelevant if the checkpoint was saved from CPU or GPU.
        checkpoint = torch.load(file_path, map_location=lambda storage, loc: storage)

        # Load remaining checkpoint information.
        self.model_name = checkpoint['model_name']
        try:
            self.optimiser = checkpoint['optimiser']
            self.logger.warning("Loaded a fully saved optimiser instead of its state dict", DeprecationWarning)
        except KeyError:
            optimiser_state_dict = checkpoint['optimiser_state_dict']
            if optimiser_state_dict is not None:
                self.set_optimiser(hparams)
                try:
                    self.optimiser.load_state_dict(optimiser_state_dict)
                except ValueError as e:
                    self.logger.warning("State dict for optimiser {} miss matches checkpoint's optimiser state dict: {}"
                                        .format(hparams.optimiser_type, e)
                                        + "\nContinuing without loading optimiser instead.")

        # Initial learning rate is required by some optimisers to compute the learning rate of the current epoch.
        if self.optimiser is not None:
            for group in self.optimiser.param_groups:
                if hasattr(group, 'initial_lr'):
                    group.setdefault('initial_lr', initial_lr)

        # Move optimiser to GPU.
        if hparams.use_gpu:
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

            if self.optimiser is not None:
                _transform_state(self.optimiser, lambda t: t.cuda(), lambda t: torch.is_tensor(t))  # Doing it manually.

        return checkpoint['epoch']

    def save_checkpoint(self, file_path, total_epoch):
        """
        Save checkpoint which consists of epoch number, model type, in/out dimensions, model state dict, whole
        optimiser, etc. Also save the EMA separately when one exists.
        """
        self.logger.info("Save checkpoint to " + file_path)
        makedirs_safe(os.path.dirname(file_path))  # Create directory if necessary.
        checkpoint_dict = {'epoch': total_epoch,
                           'model_name': self.model_name,
                           'optimiser_state_dict': self.optimiser.state_dict() if self.optimiser is not None else None,
                           # 'loss_function': loss_function
                           }
        if self.model_type:
            checkpoint_dict.update({'model_type': self.model_type,
                                    'dim_in': self.dim_in,
                                    'dim_out': self.dim_out,
                                    'model_state_dict': self.model.state_dict()})
        else:
            checkpoint_dict.update({'model': self.model})  # Special case where model_type is not given.

        torch.save(checkpoint_dict, file_path)
        # TODO: Also save the random generator states:
        #       torch.random.get_rng_state()
        #       torch.random.set_rng_state()
        #       Same for random package?
        # TODO: Save scheduler_type in checkpoint as well.

        if self.ema:
            self.save_model(file_path + "_ema", self.ema.model, self.model_type, self.dim_in, self.dim_out, verbose=True)

    def forward(self, in_tensor, hparams, batch_seq_lengths=None, target=None):
        """Forward one example through the model.

        :param in_tensor:           PyTorch tensor or numpy array.
        :param hparams:             Hyper-parameter container.
        :param batch_seq_lengths:   Tuple with the real length (not padded) of each tensor in in_tensor.
        :param target:              PyTorch tensor or numpy array, used for teacher forcing, can be None.
        :return:                    Output of model (numpy array).
        """
        self.model.eval()

        # If input tensor is numpy array convert it to torch tensor.
        if isinstance(in_tensor, np.ndarray):
            in_tensor = torch.from_numpy(in_tensor)
        if hparams.use_gpu and in_tensor is not None:
            in_tensor = in_tensor.cuda()

        # If target tensor is given and numpy array, convert it to torch tensor.
        if target is not None:
            if isinstance(target, np.ndarray):
                target = torch.from_numpy(target)
            if hparams.use_gpu:
                target = target.cuda()

        # If lengths is tuple or numpy array convert it to torch tensor.
        if batch_seq_lengths is not None:
            if isinstance(batch_seq_lengths, tuple):
                batch_seq_lengths = torch.tensor(batch_seq_lengths, dtype=torch.long)
            if isinstance(batch_seq_lengths, np.ndarray):
                batch_seq_lengths = torch.from_numpy(batch_seq_lengths, dtype=torch.long)
        else:
            batch_seq_lengths = torch.tensor([len(in_tensor)], dtype=torch.long)

        hidden = self.model.init_hidden(len(batch_seq_lengths))
        output, hidden = self.model(in_tensor, hidden, batch_seq_lengths, batch_seq_lengths[0], target)

        # Convert back (to cpu and) to numpy.
        return ModelHandlerPyTorch._return_values_to_numpy(output, hparams.use_gpu),\
               ModelHandlerPyTorch._return_values_to_numpy(hidden, hparams.use_gpu)

    @staticmethod
    def _return_values_to_numpy(return_values, from_gpu):
        """Convert all tensors in return_values to CPU. Return_values can be a tuple of tuples of tuples etc."""
        if return_values is None:
            return None

        if isinstance(return_values, tuple):
            return tuple(map(lambda x: ModelHandlerPyTorch._return_values_to_numpy(x, from_gpu), return_values))
        # TODO: Handle list and dict here as well?

        # Return value is tensor.
        if from_gpu:
            return return_values.detach().cpu().numpy()
        else:
            return return_values.detach().numpy()

    def process_dataloader(self, dataloader, loss_function, hparams, total_epoch, current_epoch=None, training=True):
        """
        Train or test the model by loading batches from the dataloader.

        :param dataloader:        Dataloader of the train/test set.
        :param loss_function:     PyTorch function/class to compute loss.
        :param hparams:           Hyper-parameter container.
        :param total_epoch:       Total number of training epochs. Used to compute the iteration for some schedulers
                                  when the learning rate is not reset in beginning of current train loop.
        :param current_epoch:     Number of epoch in current training loop. Used to compute the iteration for some
                                  schedulers when the learning rate was reset in beginning of current train loop
        :param training:          Determines if it runs the training or testing loop.
        :return:                  Tuple of total loss and total loss per output feature.
        """

        model = self.model
        if training:
            model.train()
            self.logger.info("{}: Train with {} on {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                                              self.optimiser,
                                                              str(torch.cuda.device_count()) + " GPU(s)." if hparams.use_gpu else "1 CPU."))
        else:
            if self.ema is not None:
                self.logger.info("Using averaged model for validation.")
                model = self.ema.model
            model.eval()
            self.logger.info("{}: Compute loss of validation set.".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

        if hparams.log_memory_consumption:
            self.logger.info('CPU: {:.0f} MB, GPU: {} MB'
                             .format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e3,
                                     str(get_gpu_memory_map()) if hparams.use_gpu else "-"))

        # Multi-GPU support.
        if hparams.num_gpus > 1:
            model = DataParallel(model, dim=0 if hparams.batch_first else 1)
            model.init_hidden = model.module.init_hidden  # Make the init_hidden method directly accessible.

        # Log loss after each <hparams.logging_batch_index_perc>% of batches.
        logging_batch_index = (len(dataloader) // hparams.logging_batch_index_perc) + 1

        current_batch_index = -1  # Consider special case for first batch.
        current_batch = None
        loss = None
        total_loss = 0
        loss_features = None

        # FIXME: Experimental implementation to pre-load the next batch to GPU. Does not work yet because it blocks.
        if hparams.use_gpu and hparams.preload_next_batch_to_gpu:
            # Create an iterator around the dataloader to pop the first element before the for loop.
            dataloader = iter(dataloader)
            # Pop the first batch from the dataloader.
            current_batch = next(dataloader)
            current_batch_index = 0
            # Move the first batch to GPU.
            inputs, target, seq_lengths_input, seq_lengths_target, mask, _ = current_batch
            inputs = inputs.cuda(async=hparams.dataset_load_async) if inputs is not None else None
            seq_lengths_input = seq_lengths_input.cuda(async=hparams.dataset_load_async)
            target = target.cuda(async=hparams.dataset_load_async)
            seq_lengths_target = seq_lengths_target.cuda(async=hparams.dataset_load_async)
            mask = mask.cuda(async=hparams.dataset_load_async) if mask is not None else None
            current_batch = inputs, target, seq_lengths_input, seq_lengths_target, mask, _

        # Iterate on the batches.
        for next_batch_index, next_batch in enumerate(dataloader, current_batch_index + 1):
            # Move the next batch to GPU.
            if hparams.use_gpu:
                next_inputs, next_target, next_seq_lengths_input, next_seq_lengths_target, next_mask, _ = next_batch
                next_inputs = next_inputs.cuda(async=hparams.dataset_load_async) if next_inputs is not None else None
                next_seq_lengths_input = next_seq_lengths_input.cuda(async=hparams.dataset_load_async)
                next_target = next_target.cuda(async=hparams.dataset_load_async)
                next_seq_lengths_target = next_seq_lengths_target.cuda(async=hparams.dataset_load_async)
                next_mask = next_mask.cuda(async=hparams.dataset_load_async) if next_mask is not None else None
                next_batch = next_inputs, next_target, next_seq_lengths_input, next_seq_lengths_target, next_mask, _

            # If there is no current batch either experiment is on CPU or hparams.preload_next_batch_to_gpu is False.
            # In any case use the "next" batch for the current iteration.
            if current_batch is None:
                current_batch_index = next_batch_index
                current_batch = next_batch

            # Get data and move it to gpu if necessary.
            inputs, target, seq_lengths_input, seq_lengths_target, mask, _ = current_batch
            # self.logger.info(str(torch.max(seq_lengths_input)) + " " + str(torch.max(seq_lengths_target)))

            # Request the architecture to initialise its hidden states.
            hidden = model.init_hidden(len(seq_lengths_input))

            # Forward the input through the model.
            max_length_inputs = seq_lengths_input[0]
            if hparams.use_gpu and hparams.num_gpus > 1:
                assert(len(seq_lengths_input) % torch.cuda.device_count() == 0)  # Batch is not equally dividable into the given number of GPUs.
                max_length_inputs = max_length_inputs.repeat(hparams.num_gpus)

            # Give max length because DataParallel splits the seq_lengths_input and padding will be done according to
            # the maximum length of that subset. Combining multi GPU output will fail with a size miss match.
            # https://pytorch.org/docs/stable/notes/faq.html#pack-rnn-unpack-with-data-parallelism
            if training:
                output, hidden_out = model(inputs,
                                           hidden,
                                           seq_lengths_input,
                                           max_length_inputs,
                                           target,
                                           seq_lengths_target)
            else:
                with torch.no_grad():
                    output, hidden_out = model(inputs,
                                               hidden,
                                               seq_lengths_input,
                                               max_length_inputs,
                                               target if hparams.teacher_forcing_in_test else None,
                                               seq_lengths_target)

            # Compute loss of the output.
            loss_full = loss_function(output, target)
            assert(loss_full.nelement() > 1)  # Don't reduce the loss, so that the mask can be applied. Use reduction='none' in loss function.
            if mask is not None:
                loss_full = loss_full * mask  # Don't do inplace multiplication because both tensors could be expanded.

            if hparams.loss_per_sample:
                # Average the loss on each sample of the batch and then compute the mean, which means that
                # each sample in the batch contributes equally to the loss independently from its length.

                # Automatically determine the time dimension and sum over it.
                time_dim = 0 if loss_full.shape[0] == seq_lengths_target.max() else 1
                sample_loss_features = (loss_full.sum(dim=time_dim) / seq_lengths_target.unsqueeze(-1).float()).mean(0)  # Take mean over batch dimension.
                loss = sample_loss_features.mean()
            else:
                # Default: Average the loss over all frames, then compute the mean of all loss channels.
                sample_loss_features = (loss_full.sum(dim=(0, 1)) / sum(seq_lengths_target).float())
                loss = sample_loss_features.mean()

            if torch.isnan(loss):
                self.logger.error("Loss is nan: {}".format(sample_loss_features))
                break
            # nan_or_inf = torch.isnan(loss)
            # for params in self.model.parameters():
            #     nan_or_inf |= torch.isnan(params.data).any()
            #     nan_or_inf |= (params.data == float("inf")).any()
            #     nan_or_inf |= (params.data == -float("inf")).any()
            #     if nan_or_inf:
            #         pdb.set_trace()

            if training:
                # Zero all gradients in the optimiser.
                self.optimiser.zero_grad()
                # Propagate error backwards.
                loss.backward(retain_graph=hparams.backward_retain_graph)

                # # DEBUG: Check for NaNs and Infs and start pdb debugger if some are found.
                # nan_or_inf = torch.isnan(loss)
                # for params in reversed(list(self.model.parameters())):
                #     nan_or_inf |= torch.isnan(params.grad).any()
                #     nan_or_inf |= (params.grad == float("inf")).any()
                #     nan_or_inf |= (params.grad == -float("inf")).any()
                #     if nan_or_inf:
                #         pdb.set_trace()

                if hparams.replace_inf_grads_by_zero:
                    # Replace inf/-inf in gradients by 0.
                    for params in self.model.parameters():
                        # Check for positive infinity.
                        indices = (params.grad == float("inf"))
                        if indices.any():
                            self.logger.warning("Replace inf by 0.0 in the gradient of " + params)
                            params.grad[indices] = 0.0
                        # Check for negative infinity.
                        indices = (params.grad == -float("inf"))
                        if indices.any():
                            self.logger.warning("Replace -inf by 0.0 in the gradient of " + params)
                        params.grad[indices] = 0.0

                # Clip gradients.
                if hparams.grad_clip_norm_type is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), hparams.grad_clip_max_norm, hparams.grad_clip_norm_type)  # Adds a small bias.
                if hparams.grad_clip_thresh is not None:
                    torch.nn.utils.clip_grad_value_(self.model.parameters(), hparams.grad_clip_thresh)  # Adds a big bias.

                # Change all model weights depending on their gradient.
                self.optimiser.step()

                # Update moving average.
                if self.ema:
                    self.ema.update_params(model)

                # Run the scheduler_type if one exists and should be called after some iterations.
                if self.scheduler:
                    if hparams.iterations_per_scheduler_step:
                        if hparams.use_saved_learning_rate:
                            current_iteration = (total_epoch - 1) * len(dataloader) + current_batch_index + 1
                        else:
                            current_iteration = (current_epoch - 1) * len(dataloader) + current_batch_index + 1
                        if current_iteration % hparams.iterations_per_scheduler_step == 0:
                            self._scheduler_step_fn(loss.detach(), current_iteration)
                            # self.logger.info(str(self.optimiser))

            # Logging current error.
            if current_batch_index % logging_batch_index == 0:
                self.logger.info('{} [{:{front_pad}d}/{}]\tLoss: {:.3f}'
                                 .format("Trained " if training else "Tested ",
                                         current_batch_index + 1,
                                         len(dataloader),
                                         loss,
                                         front_pad=len(str(len(dataloader))))
                                 + ("\tCPU: {:.0f} MB, GPU: {} MB"
                                    .format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e3,
                                            str(get_gpu_memory_map()) if hparams.use_gpu else "-"))
                                 if hparams.log_memory_consumption else "")

                if training:
                    fn_log_per_batch = getattr(self.model, "log_per_batch", None)
                    if callable(fn_log_per_batch):
                        fn_log_per_batch()

            loss = loss.detach()
            # Accumulate loss.
            total_loss += loss
            if loss_features is None:
                loss_features = sample_loss_features.detach()
            else:
                loss_features += sample_loss_features.detach()
                del sample_loss_features

            del inputs, target, hidden, seq_lengths_input, max_length_inputs, seq_lengths_target, mask
            del output, hidden_out, loss_full

            if hparams.use_gpu and hparams.preload_next_batch_to_gpu:
                # Use the next_batch as current_batch in next iteration.
                current_batch = next_batch
                current_batch_index = next_batch_index
            else:
                # Reset the current_batch to None so that in next iteration current_batch = next_batch again.
                current_batch = None
                # current_batch_index = None  # This is actually unnecessary.

        loss_features /= len(dataloader)
        total_loss /= len(dataloader)
        if not training:
            self.logger.info('Test set: Average loss: {:.4f}, error_features:\n{})\n'.format(total_loss, loss_features.view(1, -1)))

            fn_log_per_test = getattr(self.model, "log_per_test", None)
            if callable(fn_log_per_test):
                fn_log_per_test()

        np_total_loss = total_loss.cpu().numpy()
        np_loss_features = loss_features.cpu().numpy()
        del total_loss, loss_features

        return np_total_loss, np_loss_features

    def test(self, hparams, total_epoch, current_epoch, loss_function):
        if hparams.use_gpu:
            assert (hparams.num_gpus <= torch.cuda.device_count())  # Specified number of GPUs is incorrect.

        return self.process_dataloader(self.dataloader_val,
                                       loss_function,
                                       hparams,
                                       total_epoch,
                                       current_epoch,
                                       training=False)

    def train(self, hparams, total_epoch, current_epoch, loss_function):
        if hparams.use_gpu:
            assert (hparams.num_gpus <= torch.cuda.device_count())  # Specified number of GPUs is incorrect.

        if hparams.ema_decay and not self.ema:
            average_model = ModelFactory.create(self.model_type, self.dim_in, self.dim_out, hparams)
            average_model.load_state_dict(self.model.state_dict())
            self.ema = ExponentialMovingAverage(average_model, hparams.ema_decay)

        return self.process_dataloader(self.dataloader_train,
                                       loss_function,
                                       hparams,
                                       total_epoch,
                                       current_epoch,
                                       training=True)[0]

    def run_scheduler(self, loss, current_epoch):
        if self.scheduler is not None:
            # self.logger.info("Call scheduler.")
            self._scheduler_step_fn(loss, current_epoch)
