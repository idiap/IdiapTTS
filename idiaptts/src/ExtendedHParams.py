#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#

# System imports.
import logging
import types
import numpy as np

# Third-party imports.
from tensorflow.contrib.training import HParams
from tensorflow.contrib.training.python.training.hparam import _cast_to_type_if_compatible

# Local source tree imports.


class ExtendedHParams(HParams):

    def __setattr__(self, name, value):
        """Ensure type check for all attribute assignments."""
        if name in ["_hparam_types", "_model_structure"]:
            super().__setattr__(name, value)
        else:
            self.set_hparam(name, value)

    def set_hparam(self, name, value):
        """Override to use super().__setattr_(...) function instead to prevent an infinite loop."""
        param_type, is_list = self._hparam_types[name]
        if isinstance(value, list):
            if not is_list:
                raise ValueError('Must not pass a list for single-valued parameter: %s' % name)
            super().__setattr__(name, [self._cast_to_type_if_compatible(name, param_type, v) for v in value])
        else:
            if is_list:
                raise ValueError('Must pass a list for multi-valued parameter: %s.' % name)
            super().__setattr__(name, self._cast_to_type_if_compatible(name, param_type, value))

    @staticmethod
    def _cast_to_type_if_compatible(name, param_type, value):
        """Adding extra check for function type."""
        if param_type is types.FunctionType:
            if callable(value):
                return value
            else:
                raise ValueError('Must pass a callable object for function parameter: %s' % name)
        elif param_type is np.ndarray:
            if isinstance(value, np.ndarray):
                return value
            else:
                raise ValueError('Must pass a numpy.ndarray object for function parameter: %s' % name)
        else:
            return _cast_to_type_if_compatible(name, param_type, value)

    def setattr_no_type_check(self, name, value):
        """Function to explicitly set an attribute without checking its previous type."""
        self.del_hparam(name)
        self.add_hparam(name, value)

    def add_hparams(self, **kwarg):
        for key, value in kwarg.items():
            try:
                self.add_hparam(key, value)
            except ValueError:
                self.set_hparam(key, value)

    def override_from_hparam(self, hparam):
        for name, value in hparam.values().items():
            try:
                self.set_hparam(name, value)
            except KeyError:
                self.add_hparam(name, value)
        return self

    def verify(self):
        for attr, value in self.__dict__.items():
            if attr not in ["_hparam_types", "_model_structure"]:
                if attr not in self._hparam_types:
                    logging.warning("Attribute {} not in types dictionary."
                                    "Please use add_hparam or add_hparams to add attributes.".format(attr))

    def get_debug_string(self):
        values = self.values()
        hp = ['  %s: %s' % (name, values[name]) for name in sorted(values)]
        return 'Hyperparameters:\n' + '\n'.join(hp)

    @staticmethod
    def create_hparams(hparams_string=None, verbose=False):
        """Create model hyper-parameters. Parse non-default from given string."""

        hparams = ExtendedHParams(
            ################################
            # General Parameters           #
            ################################

            voice=None,  # Specifies a part of the dataset used.
            work_dir=None,  # Directory from where the script is running.
            data_dir=None,  # Database directory.
            logging_batch_index_perc=10,  # Percentage used from the full dataset between logging the train/test loss.
            start_with_test=True,  # Determines if the model is tested first before any training loops.
            # The computed loss is also used to identify the best model so far.
            # Therefore, if this is False and use_best_as_final_model is True
            # the best model of the current training will be saved, which possibly
            # overrides an older better model.
            log_memory_consumption=True,
            epochs_per_test=1,  # Number of training epochs before testing
            # NOTE that this includes the scheduler_type with epoch scheduling.

            networks_dir="nn",
            checkpoints_dir="checkpoints",  # Subdirectory within the networks_dir to save checkpoints.
            epochs_per_checkpoint=1,  # Number of epochs between checkpoints, 0 for no checkpoints at all.
            save_final_model=True,  # Determines if the model is saved after training.
            use_best_as_final_model=True,  # Substitutes the saved final model with the best of the current run.
            gen_figure_ext=".pdf",

            ################################
            # Experiment Parameters        #
            ################################
            epochs=0,
            test_set_perc=0.05,  # Percentage of samples taken from the given id_list in __init__ for testing.
            # Ignored when self.id_list_train is already set.
            # Note that self.id_list_test must be set then as well.
            val_set_perc=0.05,  # Percentage of samples taken from the given id_list in __init__ for validation.
            # Ignored when self.id_list_train is already set.
            # Note that self.id_list_val should be set then as well.
            seed=None,  # Used to initialize torch, numpy, and random.
            # If None, the id_list is not shuffled before taking test and validation set from it.
            # fp16_run=False,  # TODO: Not implemented.
            # distributed_run=False,  # TODO: Find out how distributed run works.
            # dist_url="file://distributed.dpt",
            # cudnn_enabled=True,
            # cudnn_benchmark=False,
            use_gpu=False,
            num_gpus=1,  # TODO: Change to num_devices.
            batch_first=False,  # Note: This might not be implemented properly everywhere.
            variable_sequence_length_train=None,  # Do samples in mini batches during training have variable length.
            variable_sequence_length_test=None,  # Do samples in mini batches during testing have variable length.
            shuffle_train_set=True,  # Shuffle in dataset to get mini batches.
            shuffle_val_set=False,  # Shuffle in dataset to get mini batches.
            batch_size_train=1,
            batch_size_test=48,
            batch_size_val=48,
            batch_size_benchmark=48,
            batch_size_synth=48,
            batch_size_gen_figure=48,
            dataset_num_workers_gpu=4,  # Number of workers used in dataset when running on GPU(s).
            dataset_num_workers_cpu=0,  # Number of workers used in dataset when running on CPU(s).
            dataset_pin_memory=True,
            dataset_load_async=True,
            teacher_forcing_in_test=False,  # If True, the targets are also given to the model when running the test
            # (needed for WaveNet).
            preload_next_batch_to_gpu=False,  # If True loads the next batch to GPU while processing the current one.
            # This enhances GPU usage for the cost of memory (two batches are loaded).
            # TODO: This does not work yet, because cuda async does lazy loading.

            ################################
            # Data Parameters              #
            ################################
            input_norm_params_file_prefix=None,
            output_norm_params_file_prefix=None,
            len_in_out_multiplier=1,
            out_dir=None,
            world_dir=None,  # Full path to directory with WORLD features, required for copy synthesis.
            # If None, hparams.out_dir/../WORLD is used.

            ################################
            # Audio Parameters             #
            ################################
            frame_size_ms=5,

            ################################
            # Model Parameters             #
            ################################
            model_type=None,
            model_name=None,
            model_path=None,  # Full path to load model from, otherwise dir_out/networks_dir/model_name.
            ignore_layers=["type dummy"],  # List of layers which are ignored when loading the model from model_path.
            # Giving the dummy ensures that hparams expects a list of strings.
            dropout=0.0,
            hidden_init=0.0,  # Hidden state init value
            train_hidden_init=False,  # Is the hidden state init value trainable  # TODO: Unused?

            ################################
            # Optimization Hyperparameters #
            ################################
            loss_per_sample=False,  # If True the loss is first averaged on each sample and then over the batch.
            # If False the loss is averaged over each frame in the whole batch (default).
            backward_retain_graph=False,  # Determines if the gradient computation should do aggressive memory freeing.
            # Only needed when gradient computational graph is reused.
            optimiser_type="Adam",  # "Adam", "SGD"  TODO: more
            optimiser_args=dict(),  # Set optimiser arguments. Preferred way to set learning rate: optimiser_args["lr"]=
            use_saved_learning_rate=True,  # Use the learning rate saved with a model after loading it.
            replace_inf_grads_by_zero=False,  # Automatically substitute +/- inf gradients with zero during training.
            # dynamic_loss_scaling=True,
            ema_decay=None,  # Any value enables EMA. EMA models are saved with a _ema in the end.

            scheduler_type="default",  # "None", "Plateau", "Exponential","Noam",  TODO: "Step", "Cyclic_cosine"
            scheduler_args=dict(),
            iterations_per_scheduler_step=None,  # Number of training iterations after which the scheduler step function
            # is called with the current loss and total number of iterations as
            # parameter. If None the scheduler is not called.
            epochs_per_scheduler_step=None,  # Number of training epochs after which the scheduler step function is
            # called with the current validation loss and total number of epochs.
            # When a model is loaded the epoch number continues from the epoch number
            # stored in the model.

            grad_clip_norm_type=None,  # If None no gradient clipping otherwise uses grad_clip_max_norm (small bias).
            grad_clip_max_norm=None,  # Ignored if grad_clip_norm_type is None.
            grad_clip_thresh=None,  # Clip absolute value of gradient (big bias).

            # Set optimiser or scheduler_type to ignore type configuration above. Used to try new implementations.
            optimiser=None,  # Will be called with model parameters only. Set other parameters with partial.
            # Example: partial(torch.optim.Adam, **args)).
            scheduler=None,  # Will be called with optimiser only. Set other parameters with partial.
            # Example: partial(ReduceLROnPlateau, **args)).

            ################################
            # Synthesis Parameters         #
            ################################
            synth_vocoder="WORLD",  # "WORLD", "r9y9wavenet_quantized_16k_world_feats"
            synth_ext="wav",  # Extension of the output audio.
            synth_fs=16000,
            sp_type="mcep",
            num_coded_sps=60,  # Number of coded spectral features.
            synth_dir=None,  # Output directory to save the synthesised audio.
            synth_acoustic_model_path=None,
            synth_file_suffix='',  # Suffix of synthesised files name.
            do_post_filtering=False,  # Merlin post-filtering of cepstrum.
            synth_gen_figure=False,  # Saves a plot when synthesising.

            # epochs_per_plot=0,  # No plots per epoch with <= 0. # TODO: plot in run method each ... epochs.
            # plot_per_epoch_id_list=None,  # TODO: Id(s) in the dictionary which are plotted.
        )
        hparams.set_hparam("ignore_layers", list())  # Remove string type dummy.

        if hparams_string:
            logging.info('Parsing command line hparams: %s', hparams_string)
            hparams.parse(hparams_string)

        if verbose:
            logging.info(hparams.get_debug_string())

        return hparams
