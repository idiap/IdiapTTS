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

# Third-party imports.
import torch
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader
from torch.nn import DataParallel

# Local source tree imports.
if not any(p.endswith("IdiapTTS") for p in sys.path):
    parent_dirs = os.path.realpath(__file__).split(os.sep)
    dir_itts = str.join(os.sep, parent_dirs[:parent_dirs.index("IdiapTTS") + 1])
    sys.path.append(dir_itts)  # Adds the IdiapTTS folder to the path, required to work on grid.
from misc.utils import get_gpu_memory_map
from src.neural_networks.ModelHandler import ModelHandler
from src.neural_networks.pytorch.models.RNNDyn import *
# from src.neural_networks.pytorch.ExponentialMovingAverage import ExponentialMovingAverage


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
    def __init__(self, hparams):
        """Default constructor. Checks CUDA support if use_gpu is True."""

        if not hasattr(hparams, "batch_size_train") or not hparams.batch_size_train > 1:
            hparams.variable_sequence_length_train = False
        if not hasattr(hparams, "batch_size_val") or not hparams.batch_size_val > 1:
            hparams.variable_sequence_length_val = False

        super().__init__()

        self.current_epoch = None
        self.start_epoch = None

        self._scheduler_step_fn = None
        self.ema = None  # Exponential moving average object.

        # Register different architectures.
        self.register_architecture(RNNDyn)
        self.register_architecture(MerlinAcoustic)
        self.register_architecture(Interspeech18baseline)
        self.register_architecture(BaselineRNN_Yamagishi)
        self.register_architecture(Icassp19baseline)

        # Register optional architectures.
        requirement_warping_layer = importlib.util.find_spec("WarpingLayer")
        if requirement_warping_layer:
            from src.neural_networks.pytorch.models.WarpingLayer import WarpingLayer
            self.register_architecture(WarpingLayer)

        requirement_neuralfilters = importlib.util.find_spec("neural_filters")
        if requirement_neuralfilters:
            from src.neural_networks.pytorch.models.NeuralFilters import NeuralFilters
            from src.neural_networks.pytorch.models.PhraseNeuralFilters import PhraseNeuralFilters
            self.register_architecture(PhraseNeuralFilters)
            self.register_architecture(NeuralFilters)

        # requirement_wavenet_vocoder = importlib.util.find_spec("wavenet_vocoder")
        # if requirement_wavenet_vocoder:
        #     from src.neural_networks.pytorch.models.WaveNetWrapper import WaveNetWrapper
        #     self.register_architecture(WaveNetWrapper)

        requirement_nvtacotron2 = importlib.util.find_spec("tools.tacotron2")
        if requirement_nvtacotron2:
            from src.neural_networks.pytorch.models.NVTacotron2Wrapper import NVTacotron2Wrapper
            self.register_architecture(NVTacotron2Wrapper)

    @staticmethod
    def cuda_is_available():
        return torch.cuda.is_available()

    @staticmethod
    def device_count():
        return torch.cuda.device_count()

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
        dataloader_train = DataLoader(dataset=dataset_train,
                                      batch_size=hparams.batch_size_train,
                                      shuffle=hparams.shuffle_train_set,
                                      num_workers=hparams.dataset_num_workers_gpu if hparams.use_gpu else hparams.dataset_num_workers_cpu,
                                      collate_fn=partial(self.prepare_batch if collate_fn is None else collate_fn,
                                                         common_divisor=common_divisor,
                                                         batch_first=hparams.batch_first),
                                      pin_memory=hparams.dataset_pin_memory)
        dataloader_val = DataLoader(dataset_val,
                                     batch_size=hparams.batch_size_val,  # Used to be batch_size_test, please change it in your My* class.
                                     shuffle=hparams.shuffle_val_set,
                                     num_workers=hparams.dataset_num_workers_gpu if hparams.use_gpu else hparams.dataset_num_workers_cpu,
                                     collate_fn=partial(self.prepare_batch if collate_fn is None else collate_fn,
                                                        common_divisor=common_divisor,
                                                        batch_first=hparams.batch_first),
                                     pin_memory=hparams.dataset_pin_memory)
        self.set_dataloaders(dataloader_train, dataloader_val)

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
            assert(hparams.scheduler_type != "default")  # Please define a default scheduler type in the main trainer class.

            if hparams.scheduler_type == "Plateau":
                self.scheduler = ReduceLROnPlateau(self.optimiser, **hparams.scheduler_args)
                self._scheduler_step_fn = self._scheduler_step_with_loss
                if hparams.epochs_per_scheduler_step is None and hparams.iterations_per_scheduler_step is None:
                    hparams.epochs_per_scheduler_step = 1
                return

            elif hparams.scheduler_type == "Exponential":
                self.scheduler = ExponentialLR(self.optimiser, last_epoch=current_epoch - 1, **hparams.scheduler_args)
                self._scheduler_step_fn = self._scheduler_step
                if hparams.epochs_per_scheduler_step is None and hparams.iterations_per_scheduler_step is None:
                    hparams.iterations_per_scheduler_step = 1
                return

            elif hparams.scheduler_type == "Noam":
                if "wormup_steps" not in hparams.scheduler_args:
                    self.logger.error("Please define wormup_steps in hparams.scheduler_args.")
                self.scheduler = LambdaLR(self.optimiser, lambda iteration: float(hparams.scheduler_args['wormup_steps'])**0.5 * np.minimum((iteration + 1) * float(hparams.scheduler_args['wormup_steps'])**-1.5, (iteration + 1)**-0.5))
                self._scheduler_step_fn = self._scheduler_step
                if hparams.epochs_per_scheduler_step is None and hparams.iterations_per_scheduler_step is None:
                    hparams.iterations_per_scheduler_step = 1
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

        self.current_epoch = 0
        self.start_epoch = 0

        architecture_found = False
        for architecture in self.registered_architectures:
            if re.match(architecture.IDENTIFIER, hparams.model_type) is not None:
                self.model = architecture(dim_in, dim_out, hparams)
                self.logger.info("Create network of type: " + hparams.model_type)
                if hparams.model_name is not None:
                    self.logger.info("Selected network name: " + hparams.model_name)
                architecture_found = True
                break

        if not architecture_found:
            raise TypeError("Unkown network type: " + hparams.model_type + ". No model was created.")

        # Send model to gpu, if requested.
        if hparams.use_gpu:
            self.logger.info("Convert network to GPU.")
            self.model = self.model.cuda()

    def load_model(self, file_path, use_gpu, initial_lr=0.0):
        """
        Load a model by name, also transfers to GPU if requested in constructor.

        :param file_path:         Full path to checkpoint.
        :param use_gpu:           Convert model to GPU if true.
        :param initial_lr:        Initial learning rate of the model. Required by some schedulers to compute the
                                  learning rate of the current epoch/iteration.
        :return:                  The loaded model.
        """

        # The lambda expression makes it irrelevant if the checkpoint was saved from CPU or GPU.
        checkpoint = torch.load(file_path, map_location=lambda storage, loc: storage)
        self.logger.info("Load model from " + file_path)
        self.current_epoch = checkpoint['epoch']
        self.start_epoch = self.current_epoch
        self.model = checkpoint['model']
        self.model_name = checkpoint['model_name']
        self.model.set_gpu_flag(use_gpu)
        self.optimiser = checkpoint['optimiser']
        if self.optimiser is not None:
            for group in self.optimiser.param_groups:
                group.setdefault('initial_lr', initial_lr)
        if use_gpu:
            self.model.cuda()

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

        return self.model

    def save_model(self, file_path):
        """Save epoch number, the whole model, and whole optimiser."""
        self.logger.info("Save model to " + file_path)
        torch.save({'epoch': self.current_epoch,
                    'model': self.model,
                    'model_name': self.model_name,
                    'optimiser': self.optimiser},
                   file_path)
        # TODO: Also save the random generator states:
        #       torch.random.get_rng_state()
        #       torch.random.set_rng_state()
        #       Same for random package?
        # TODO: Save scheduler_type in checkpoint as well.

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

    def process_dataloader(self, dataloader, hparams, training=True):
        """
        Train or test the model by loading batches from the dataloader.

        :param dataloader:        Dataloader of the train/test set.
        :param hparams:           Hyper-parameter container.
        :param training:          Determines if it runs the training or testing loop.
        :return:                  Tuple of total loss and total loss per output feature.
        """

        model = self.model
        if training:
            model.train()
            self.logger.info("{}: Train with {} on {}".format(datetime.now(),
                                                              self.optimiser,
                                                              str(torch.cuda.device_count()) + " GPU(s)." if hparams.use_gpu else "1 CPU."))
        else:
            self.logger.info(str(datetime.now()) + ": Compute loss of validation set.")
            if self.ema is not None:
                self.logger.info("Using averaged model for validation.")
                model = self.ema.get_averaged_model(self.model)
            model.eval()

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
            loss_full = self.loss_function(output, target)
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

            # if torch.isnan(loss):
            #     break
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
                if self.ema is not None:
                    for name, param in model.named_parameters():
                        if name in self.ema.shadow:
                            self.ema.update(name, param.data)

                # Run the scheduler_type if one exists and should be called after some iterations.
                if self.scheduler is not None:
                    if hparams.iterations_per_scheduler_step is not None and (current_batch_index + 1) % hparams.iterations_per_scheduler_step == 0:
                        self._scheduler_step_fn(loss.detach(), (self.start_epoch + self.current_epoch - 1) * len(dataloader) + current_batch_index + 1)
                        # self.logger.info(str(self.optimiser))

            # Logging current error.
            if current_batch_index % logging_batch_index == 0:
                if training:
                    self.logger.info('Trained: {}/{} [{:{front_pad}d}/{}]\tLoss: {:.3f}'
                                     .format(self.current_epoch,
                                             self.start_epoch + hparams.epochs,
                                             current_batch_index + 1,
                                             len(dataloader),
                                             loss,
                                             front_pad=len(str(len(dataloader))))
                                     + ("\tCPU: {:.0f} MB, GPU: {} MB"
                                        .format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e3,
                                                str(get_gpu_memory_map()) if hparams.use_gpu else "-"))
                                     if hparams.log_memory_consumption else "")
                else:
                    self.logger.info('Tested: [{:{front_pad}d}/{}]\tLoss: {:.3f}'
                                     .format(current_batch_index + 1,
                                             len(dataloader),
                                             loss,
                                             front_pad=len(str(len(dataloader))))
                                     + ("\tCPU: {:.0f} MB, GPU: {} MB"
                                        .format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e3,
                                                str(get_gpu_memory_map()) if hparams.use_gpu else "-")
                                        if hparams.log_memory_consumption else ""))

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

    def run(self, hparams, loss_function=None):
        """
        Run train followed by test method for the number of times specified in epochs.

        :param hparams:           Hyper-parameter container. Specifies epochs in hparams.epochs.
        :param loss_function:     Explicit loss function, otherwise self.loss_function is used.
        :return:                  List containing the loss of each epoch.
        """
        if hparams.use_gpu:
            assert(hparams.num_gpus <= torch.cuda.device_count())  # Specified number of GPUs is incorrect.

        # Only override loss function if not loaded from checkpoint.
        if self.loss_function is None:
            assert(loss_function is not None)  # Please set self.loss_function in the trainer construction.
            self.loss_function = loss_function
        if hparams.use_gpu:
            self.loss_function = loss_function.cuda()

        self.set_optimiser(hparams)
        self.set_scheduler(hparams, self.current_epoch)

        all_loss = list()  # List which is returned, containing all loss so that progress is visible.
        all_loss_train = list()
        best_loss = np.nan

        # Compute error before first iteration.
        if hparams.start_with_test:
            loss, loss_features = self.process_dataloader(self.dataloader_val, hparams, training=False)
            all_loss_train.append(-1.0)  # Set a placeholder at the train losses.
            all_loss.append(loss)
            best_loss = loss  # Variable to save the current best loss.

        # Add an ExponentialMovingAverage object if requested.
        if hparams.exponential_moving_average:
            raise NotImplementedError()
            # self.ema = ExponentialMovingAverage(hparams.exponential_moving_average_decay)
            # for name, param in self.model.named_parameters():
            #     if param.requires_grad:
            #         self.ema.register(name, param.data)

        # For each epoch do...
        for epoch in range(0, hparams.epochs):
            # Increment epoch number.
            self.current_epoch += 1

            # Train one epoch.
            all_loss_train.append(self.process_dataloader(self.dataloader_train, hparams)[0])

            if epoch % hparams.epochs_per_test == 0:
                # Compute error on validation set.
                loss, loss_features = self.process_dataloader(self.dataloader_val, hparams, training=False)

                # Save loss in a list which is returned.
                all_loss.append(loss)

                # Stop when loss is NaN. Reloading from checkpoint if necessary.
                if np.isnan(loss):
                    return

                # Save checkpoint if path is given.
                if hparams.out_dir is not None:
                    path_checkpoint = os.path.join(hparams.out_dir, hparams.networks_dir, hparams.checkpoints_dir)
                    # Check when to save a checkpoint.
                    if hparams.epochs_per_checkpoint > 0 and epoch % hparams.epochs_per_checkpoint == 0:
                        self.save_model(os.path.join(path_checkpoint, hparams.model_name + "-e" + str(self.current_epoch) + '-' + str(self.loss_function)))
                    # Always save best checkpoint with special name.
                    if loss < best_loss or np.isnan(best_loss):
                        best_loss = loss
                        self.save_model(os.path.join(path_checkpoint, hparams.model_name + "-best"))

                # Run the scheduler_type if one exists.
                if self.scheduler is not None:
                    if hparams.epochs_per_scheduler_step is not None and self.current_epoch % hparams.epochs_per_scheduler_step == 0:
                        # self.logger.info("Call scheduler.")
                        self._scheduler_step_fn(loss, self.current_epoch + 1)

        return all_loss, all_loss_train
