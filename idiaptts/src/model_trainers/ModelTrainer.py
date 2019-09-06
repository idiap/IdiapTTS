#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#


# System imports.
import copy
import logging
import math
import os
import resource
import sys
from datetime import timedelta
from functools import partial
from operator import itemgetter
from timeit import default_timer as timer
import numpy as np
import soundfile
from datetime import datetime
import random
import platform
from shutil import copy2

# Third-party imports.
import pydub
from pydub import AudioSegment
import pyworld
import pysptk
from nnmnkwii.postfilters import merlin_post_filter

# Local source tree imports.
from idiaptts.src.ExtendedHParams import ExtendedHParams
from idiaptts.src.data_preparation.world.WorldFeatLabelGen import WorldFeatLabelGen
# from idiaptts.src.data_preparation.audio.RawWaveformLabelGen import RawWaveformLabelGen
from idiaptts.src.neural_networks.pytorch.ModelHandlerPyTorch import ModelHandlerPyTorch
from idiaptts.misc.utils import makedirs_safe, get_gpu_memory_map, sample_linearly


class ModelTrainer(object):
    """
    Baseclass for all trainers.

    Load input and output data by generators (set by subclass). Perform normalisation and length mismatch fix.
    Select 5% of data (at least one) for testset. The model_handler implementation of a framework is used as interface.
    Subclasses have to set up the data and synthesize attributes by instantiating generators and possibly overwriting
    the synthesize method.
    """
    logger = logging.getLogger(__name__)
    dir_extracted_acoustic_features = "../WORLD/"  # Default is hparams.out_dir/self.dir_extracted_acoustic_features,
                                                   # but can be overwritten by hparams.world_dir.

    def __init__(self, id_list, hparams):
        """Default constructor.

        :param id_list:              List or tuple of ids as strings. This list is separated into evaluation, test, and training set.
        :param hparams:              An object holding all hyper parameters.
        """

        self.logger.info("Running on host {}.".format(platform.node()))

        assert(hparams is not None)

        if not hasattr(hparams, "batch_size_train") or not hparams.batch_size_train > 1:
            hparams.variable_sequence_length_train = False
        if not hasattr(hparams, "batch_size_val") or not hparams.batch_size_val > 1:
            hparams.variable_sequence_length_val = False

        if hparams.use_gpu:
            if hparams.num_gpus > 1:
                os.environ['CUDA_VISIBLE_DEVICES'] = str(tuple(range(hparams.num_gpus)))
            if ModelHandlerPyTorch.cuda_is_available():
                device_count = ModelHandlerPyTorch.device_count()
                if not device_count == hparams.num_gpus:
                    self.logger.error("Specified GPU count in hparams.num_gpus ({}) doesn't match hardware ({}).".format(hparams.num_gpus, device_count))
                assert(device_count == hparams.num_gpus)  # Specified GPU count doesn't match hardware.
            else:
                self.logger.warning("No CUDA device available, use CPU mode instead.")
                hparams.use_gpu = False

        if "lr" not in hparams.optimiser_args and hasattr(hparams, "learning_rate") and hparams.learning_rate is not None:  # Backwards compatibility.
            hparams.optimiser_args["lr"] = hparams.learning_rate

        if hparams.seed is not None:
            ModelHandlerPyTorch.seed(hparams.seed)  # Seed the backend.
            np.random.seed(hparams.seed)
            random.seed(hparams.seed)

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
            self.id_list_train = id_list_shuffled[num_valset:-num_testset] if num_testset > 0\
                                                                           else id_list_shuffled[num_valset:]
            assert(len(self.id_list_train) > 0)

        # Create and initialize model.
        self.logger.info("Create ModelHandler.")
        self.model_handler = ModelHandlerPyTorch()  # A handler for the NN models depending on the NN frameworks.

        # Data attributes.
        self.InputGen = None  # Used in the datasets.
        self.OutputGen = None  # Used in the datasets.
        self.dataset_train = None
        self.dataset_val = None

        self.batch_collate_fn = None  # Function used to combine samples to one batch.
        self.batch_decollate_fn = None  # Function used to split the batched output of a model.
                                        # Result is directly given to the gen_figure function.
                                        # Only the first element of the result is given to the post-processing function
                                        # of the OutputGen when result is a tuple or list.

        if not hasattr(self, "loss_function"):  # Could have been set already in constructor of child class.
            self.loss_function = None  # Has to be defined by subclass.

        self.total_epoch = None  # Total number of epochs the current model was trained.

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
            logging_batch_index_perc=10,  # Percentage of samples used from the full dataset between logging the loss for training and testing.
            start_with_test=True,  # Determines if the model is tested first before any training loops.
                                   # The computed loss is also used to identify the best model so far.
                                   # Therefore, if this is False and use_best_as_final_model is True
                                   # the best model of the current training will be saved, which possibly
                                   # overrides an older better model.
            log_memory_consumption=True,
            epochs_per_test=1,  # Number of training epochs before testing (NOTE that this includes the scheduler_type with epoch scheduling).

            networks_dir="nn",
            checkpoints_dir="checkpoints", # Subdirectory within the networks_dir to save checkpoints.
            epochs_per_checkpoint=1,  # Number of epochs between checkpoints, 0 for no checkpoints at all.
            save_final_model=True,  # Determines if the model is saved after training.
            use_best_as_final_model=True,  # Substitutes the saved final model with the best of the current run.

            ################################
            # Experiment Parameters        #
            ################################
            epochs=0,
            test_set_perc=0.05,  # Percentage of samples taken from the given id_list in __init__ for testing.
                                 # Ignored when self.id_list_train is already set. Note that self.id_list_test must be set then as well.
            val_set_perc=0.05,   # Percentage of samples taken from the given id_list in __init__ for validation.
                                 # Ignored when self.id_list_train is already set. Note that self.id_list_val should be set then as well.
            seed=None,  # Used to initialize torch, numpy, and random. If None, the id_list is not shuffled before taking test and validation set from it.
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
            teacher_forcing_in_test=False,  # If True, the targets are also given to the model when running the test (needed for WaveNet).
            preload_next_batch_to_gpu=False,  # If True loads the next batch to GPU while processing the current one.
                                              # This enhances GPU usage for the cost of memory, because two batches are loaded to the GPU.
                                              # TODO: This does not work yet, because cuda async does lazy loading.

            ################################
            # Data Parameters              #
            ################################
            input_norm_params_file_prefix=None,
            output_norm_params_file_prefix=None,
            len_in_out_multiplier=1,
            out_dir=None,
            world_dir=None,  # Full path to directory with WORLD features, required to synthesise with extracted features.
                             # If None, hparams.out_dir/../WORLD is used.

            ################################
            # Audio Parameters             #
            ################################
            frame_size_ms=5,
            # max_wav_value=32768.0,

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
            optimiser_args=dict(),  # Set optimiser arguments. Preferred way to set learning rate: optimiser_args["lr"]=...
            use_saved_learning_rate=True,  # Use the learning rate saved with a model after loading it.
            replace_inf_grads_by_zero=False,  # Automatically substitute +/- inf gradients with zero during training.
            # dynamic_loss_scaling=True,
            ema_decay=None,  # Any value enables EMA. EMA models are saved with a _ema in the end.

            scheduler_type="default",  # "None", "Plateau", "Exponential","Noam",  TODO: "Step", "Cyclic_cosine"
            scheduler_args=dict(),
            iterations_per_scheduler_step=None,  # Number of training iterations after which the scheduler step function
                                                 # is called with the current loss and total number of iterations as parameter.
                                                 # If None the scheduler is not called.
            epochs_per_scheduler_step=None,  # Number of training epochs after which the scheduler step function is
                                             # called with the current validation loss and total number of epochs.
                                             # When a model is loaded the epoch number continues from the epoch number stored in the model.

            grad_clip_norm_type=None,  # If None no gradient clipping otherwise uses grad_clip_max_norm (small bias).
            grad_clip_max_norm=None,  # Ignored if grad_clip_norm_type is None.
            grad_clip_thresh=None,  # Clip absolute value of gradient (big bias).

            # Set optimiser or scheduler_type to ignore type configuration above. Used to try new implementations.
            optimiser=None,  # Will be called with model parameters only. Set other parameters with partial. Example: partial(torch.optim.Adam, **args)).
            scheduler=None,  # Will be called with optimiser only. Set other parameters with partial. Example: partial(ReduceLROnPlateau, **args)).

            ################################
            # Synthesis Parameters         #
            ################################
            synth_vocoder="WORLD",  # "WORLD", "r9y9wavenet_quantized_16k_world_feats"
            synth_ext="wav",  # Extension of the output audio.
            synth_fs=16000,
            num_coded_sps=60,  # Number of spectral features, currently always MGC.
            synth_dir=None,
            synth_acoustic_model_path=None,
            synth_file_suffix='',
            do_post_filtering=False,  # TODO: Merlin does some filtering before calling its vocoder. Possible implementation: https://github.com/r9y9/nnmnkwii/blob/master/nnmnkwii/postfilters/__init__.py
            synth_gen_figure=False,
            gen_figure_ext=".pdf",
            # epochs_per_plot=0,  # No plots per epoch with <= 0. # TODO: plot in run method each ... epochs.
            # plot_per_epoch_id_list=None,  # TODO: Id(s) in the dictionary which are plotted.
        )
        hparams.set_hparam("ignore_layers", list())  # Remove string type dummy.

        if hparams_string:
            logging.info('Parsing command line hparams: %s', hparams_string)
            hparams.parse(hparams_string)

        if verbose:
            logging.info('Final parsed hparams: %s', hparams.values())

        return hparams

    # def plot_outputs(self, max_epochs, id_name, outputs, target):
    #     plotter = DataPlotter()
    #     net_name = os.path.basename(self.model_handler.model_name)
    #     filename = str(os.path.join(self.out_dir, id_name + '.' + net_name))
    #     plotter.set_title(id_name + " - " + net_name)
    #
    #     # Create a plot for every dimension of outputs with its target.
    #     graphs_o = [None] * target.shape[1]
    #     graphs_t = [None] * target.shape[1]
    #     for out_idx in range(0, target.shape[1]):
    #         graphs_o[out_idx] = list()
    #         # Add all outputs to the plot.
    #         for idx, o in enumerate(outputs):
    #             # Handle special case where NN output has only one dimension.
    #             if len(o.shape) == 1:
    #                 o = o.reshape(-1, 1)
    #             graphs_o[out_idx].append((o[:, out_idx], 'e' + str(min(max_epochs, (idx + 1) * self.epochs_per_plot))))
    #         # Give data to plotter and leave two grid position for each output dimension (used for output and target).
    #         plotter.set_data_list(grid_idx=out_idx * 2, data_list=graphs_o[out_idx])
    #
    #         # Add target belonging to the output dimension.
    #         graphs_t[out_idx] = list()
    #         graphs_t[out_idx].append((target[:, out_idx], 'target[' + str(out_idx) + ']'))
    #         plotter.set_data_list(grid_idx=out_idx * 2 + 1, data_list=graphs_t[out_idx])
    #
    #     # Set label for all.
    #     plotter.set_label(xlabel='frames', ylabel='amp')
    #
    #     # Generate and save the plot.
    #     plotter.gen_plot()
    #     plotter.save_to_file(filename + ".OUTPUTS.png")
    #
    #     plotter.plt.show()

    def init(self, hparams):

        self.logger.info("CPU memory: " + str(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e3) + " MB.")
        if hparams.use_gpu:
            self.logger.info("GPU memory: " + str(get_gpu_memory_map()) + " MB.")

        # Create the necessary directories.
        makedirs_safe(os.path.join(hparams.out_dir, hparams.networks_dir, hparams.checkpoints_dir))

        # Create the default model path if not set or retrieve the name from the given path.
        if hparams.model_path is None:
            assert(hparams.model_name is not None)  # A model_path or model_name has to be given. No default exists.
            hparams.model_path = os.path.join(hparams.out_dir, hparams.networks_dir, hparams.model_name)
        elif hparams.model_name is None:
            hparams.model_name = os.path.basename(hparams.model_path)

        model_path_out = os.path.join(hparams.out_dir, hparams.networks_dir, hparams.model_name)
        if hparams.epochs <= 0:
            # Try to load the model. If it doesn't exist, create a new one and save it.
            # Return the loaded/created model, because no training was requested.
            try:
                self.total_epoch = self.model_handler.load_checkpoint(hparams.model_path,
                                                                      hparams,
                                                                      hparams.optimiser_args["lr"] if hasattr(hparams, "optimiser_args")
                                                                                                     and "lr" in hparams.optimiser_args
                                                                                                     else 0.0)
            except FileNotFoundError:
                if hparams.model_type is None:
                    self.logger.error("Model does not exist at {} and you didn't give model_type to create a new one.".format(hparams.model_path))
                    raise  # This will rethrow the last exception.
                else:
                    self.logger.warning('Model does not exist at {}. Creating a new one instead and saving it.'.format(hparams.model_path))
                    dim_in, dim_out = self.dataset_train.get_dims()
                    self.model_handler.create_model(hparams, dim_in, dim_out)
                    self.total_epoch = 0
                    self.model_handler.save_checkpoint(model_path_out, self.total_epoch)

            self.logger.info("Model ready.")
            return

        if hparams.model_type is None:
            self.total_epoch = self.model_handler.load_checkpoint(hparams.model_path,
                                                                  hparams,
                                                                  hparams.optimiser_args["lr"] if hasattr(hparams, "optimiser_args")
                                                                                                 and "lr" in hparams.optimiser_args
                                                                                                 else 0.0)
        else:
            dim_in, dim_out = self.dataset_train.get_dims()
            self.model_handler.create_model(hparams, dim_in, dim_out)
            self.total_epoch = 0

        self.logger.info("Model ready.")

    def train(self, hparams):
        """
        Train the model. Use generators for data preparation and model_handler for access.
        Generators have to be set in constructor of subclasses.

        :param hparams:          Hyper-parameter container.
        :return:                 A tuple of (all test loss, all training loss, the model_handler object).
        """

        hparams.verify()  # Verify that attributes were added correctly, print warning for wrongly initialized ones.
        self.logger.info('Final parsed hparams: %s', hparams.values())

        assert(self.model_handler)  # The init function has be called before training.

        # Skip training if epochs is not greater 0.
        if hparams.epochs <= 0:
            self.logger.info("Number of training epochs is {}. Skipping training.".format(hparams.epochs))
            return list(), list(), self.model_handler

        # Log evaluation ids.
        if len(self.id_list_val) > 0:
            valset_keys = sorted(self.id_list_val)
            self.logger.info("Validation set (" + str(len(valset_keys)) + "): "
                             + " ".join([os.path.join(os.path.split(os.path.dirname(id_name))[-1],
                                                      os.path.splitext(os.path.basename(id_name))[0]) for id_name in valset_keys]))
        # Log test ids.
        testset_keys = sorted(self.id_list_test)
        self.logger.info("Test set (" + str(len(testset_keys)) + "): "
                         + " ".join([os.path.join(os.path.split(os.path.dirname(id_name))[-1],
                                                  os.path.splitext(os.path.basename(id_name))[0]) for id_name in testset_keys]))

        # Setup the dataloaders.
        self.model_handler.set_dataset(hparams, self.dataset_train, self.dataset_val, self.batch_collate_fn)

        self.logger.info("CPU memory: " + str(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e3) + " MB.")
        if hparams.use_gpu:
            self.logger.info("GPU memory: " + str(get_gpu_memory_map()) + " MB.")

        # Run model.
        # if self.epochs_per_plot > 0:
        #     outputs = list()
        #     num_iterations = int(math.ceil(float(epochs) / float(self.epochs_per_plot)))
        #     epochs_per_iter = min(epochs, self.epochs_per_plot)
        #     for e in range(num_iterations):
        #         epochs_this_iter = min(epochs_per_iter, epochs - e * epochs_per_iter)
        #         nn_model.run(epochs_this_iter, e * epochs_per_iter)
        #         outputs.append(nn_model.forward(dict_input_labels[self.plot_per_epoch_id_name]))
        #     self.plot_outputs(epochs, self.plot_per_epoch_id_name, outputs, dict_output_labels[self.plot_per_epoch_id_name])

        # Some sanity checks.
        if hparams.epochs_per_scheduler_step:
            if hparams.epochs_per_test > hparams.epochs_per_scheduler_step:
                self.logger.warning("Model is validated only every {} epochs, ".format(hparams.epochs_per_test) +
                                    "but scheduler is supposed to run every {} epochs.".format(hparams.epochs_per_scheduler_step))
            if hparams.epochs_per_test % hparams.epochs_per_scheduler_step != 0:
                self.logger.warning("hparams.epochs_per_test ({}) % hparams.epochs_per_scheduler_step ({}) != 0. "
                                    .format(hparams.epochs_per_test, hparams.epochs_per_scheduler_step) +
                                    "Note that the scheduler is only run when current_epoch % " +
                                    "hparams.epochs_per_scheduler_step == 0. Therefore hparams.epochs_per_scheduler_step " +
                                    "should be a factor of hparams.epochs_per_test.")

        t_start = timer()
        self.logger.info('Start training: ' + str(datetime.now()))

        self.model_handler.set_optimiser(hparams)
        self.model_handler.set_scheduler(hparams, self.total_epoch if hparams.use_saved_learning_rate else 0)

        assert(self.loss_function)  # Please set self.loss_function in the trainer construction.
        loss_function = self.loss_function.cuda() if hparams.use_gpu else self.loss_function

        all_loss = list()  # List which is returned, containing all loss so that progress is visible.
        all_loss_train = list()
        best_loss = np.nan
        start_epoch = self.total_epoch

        # Compute error before first iteration.
        if hparams.start_with_test:
            self.logger.info('Test epoch [{}/{}]:'.format(start_epoch, start_epoch + hparams.epochs))
            loss, loss_features = self.model_handler.test(hparams, start_epoch, start_epoch, loss_function)
            all_loss_train.append(-1.0)  # Set a placeholder at the train losses.
            all_loss.append(loss)
            best_loss = loss  # Variable to save the current best loss.

        for current_epoch in range(1, hparams.epochs + 1):
            # Increment epoch number.
            self.total_epoch += 1

            # Train one epoch.
            self.logger.info('Train epoch [{}/{}]:'.format(self.total_epoch, start_epoch + hparams.epochs))
            train_loss = self.model_handler.train(hparams, self.total_epoch, current_epoch, loss_function)
            all_loss_train.append(train_loss)
            if np.isnan(train_loss):
                break

            # Test if requested.
            if self.total_epoch % hparams.epochs_per_test == 0:
                self.logger.info('Test epoch [{}/{}]:'.format(self.total_epoch, start_epoch + hparams.epochs))
                # Compute error on validation set.
                loss, loss_features = self.model_handler.test(hparams, self.total_epoch, current_epoch, loss_function)

                # Save loss in a list which is returned.
                all_loss.append(loss)

                # Stop when loss is NaN. Reloading from checkpoint if necessary.
                if np.isnan(loss):
                    break

                # Save checkpoint if path is given.
                if hparams.out_dir is not None:
                    path_checkpoint = os.path.join(hparams.out_dir, hparams.networks_dir, hparams.checkpoints_dir)
                    # Check when to save a checkpoint.
                    if hparams.epochs_per_checkpoint > 0 and self.total_epoch % hparams.epochs_per_checkpoint == 0:
                        model_name = "{}-e{}-{}".format(hparams.model_name, self.total_epoch, loss_function)
                        self.model_handler.save_checkpoint(os.path.join(path_checkpoint, model_name), self.total_epoch)
                    # Always save best checkpoint with special name.
                    if loss < best_loss or np.isnan(best_loss):
                        best_loss = loss
                        model_name = hparams.model_name + "-best"
                        self.model_handler.save_checkpoint(os.path.join(path_checkpoint, model_name), self.total_epoch)

                # Run the scheduler if requested.
                if hparams.epochs_per_scheduler_step:
                    if (self.total_epoch if hparams.use_saved_learning_rate else current_epoch)\
                            % hparams.epochs_per_scheduler_step == 0:
                        self.model_handler.run_scheduler(loss, self.total_epoch + 1)

        t_training = timer() - t_start
        self.logger.info('Training time: ' + str(timedelta(seconds=t_training)))
        self.logger.info('Loss progress: ' + ', '.join('{:.4f}'.format(l) for l in all_loss))
        self.logger.info('Loss train progress: ' + ', '.join('{:.4f}'.format(l) for l in all_loss_train))

        if hparams.out_dir is not None:
            # Check if best model should be used as final model. Only possible when it was save in out_dir.
            if hparams.use_best_as_final_model:
                best_model_path = os.path.join(hparams.out_dir, hparams.networks_dir, hparams.checkpoints_dir, hparams.model_name + "-best")
                try:
                    self.total_epoch = self.model_handler.load_checkpoint(best_model_path,
                                                                          hparams,
                                                                          hparams.optimiser_args["lr"] if hparams.optimiser_args["lr"]
                                                                                                         else hparams.learning_rate)
                    if self.model_handler.ema:  # EMA model should be used as best model.
                        self.model_handler.model = self.model_handler.ema.model
                        self.model_handler.ema = None  # Reset this one so that a new one is created for further training.
                        self.logger.info("Using best EMA model (epoch {}) as final model.".format(self.total_epoch))
                    else:
                        self.logger.info("Using best (epoch {}) as final model.".format(self.total_epoch))
                except FileNotFoundError:
                    self.logger.warning("No best model exists yet. Continue with the current one.")

            # Save the model if requested.
            if hparams.save_final_model:
                self.model_handler.save_checkpoint(os.path.join(hparams.out_dir, hparams.networks_dir, hparams.model_name), self.total_epoch)

        return all_loss, all_loss_train, self.model_handler

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
        raise ValueError("Unkown input {} of type {}.".format(input, type(input)))

    @staticmethod
    def split_batch(output, hidden, seq_length_output=None, permutation=None, batch_first=False):
        """
        Retrieve output and hidden from batch.

        :param output:             Batched output tensor given by network.
        :param hidden:             Batched hidden tensor given by network.
        :param seq_length_output:  Tuple containing the lengths of all samples in the batch.
        :param permutation:        Permutations previously applied to the batch, which are reverted here.
        :param batch_first:        Batch dimension is first in output.
        :return:                   List of outputs and list of hidden, where each entry corresponds to one sample in the batch.
        """

        # Split the output of the batch.
        return ModelTrainer._split_return_values(output, seq_length_output, permutation, batch_first),\
               ModelTrainer._split_return_values(hidden, seq_length_output, permutation, batch_first)

    @classmethod
    def _split_return_values(cls, input_values, seq_length_output, permutation, batch_first):
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

    def forward(self, hparams, ids_input):
        """
        Forward all given ids through the network in batches of hparams.batch_size_val.

        :param hparams:        Hyper-parameter container.
        :param ids_input:      Can be full path to file with ids, list of ids, or one id.or None.
        :return:               (Dictionary of network outputs, dictionary of post-processed (by self.OutputGen) network outputs)
        """
        assert(self.model_handler is not None)  # Check if trainer.init() was called before.
        id_list = ModelTrainer._input_to_str_list(ids_input)

        self.logger.info("Start forwarding [{0}]".format(", ".join(str(i) for i in id_list)))
        t_start = timer()
        model_output, model_output_post = self._forward_batched(hparams,
                                                                id_list,
                                                                hparams.batch_size_val,
                                                                load_target=False,
                                                                synth=False,
                                                                benchmark=False,
                                                                gen_figure=False)
        t_training = timer() - t_start
        self.logger.info('Forwarding time for {} sample(s): {}'.format(len(id_list), timedelta(seconds=t_training)))

        return model_output, model_output_post

    def synth(self, hparams, ids_input):
        """
        Synthesise all given ids with the self.synthesize function.

        :param hparams:        Hyper-parameter container.
        :param ids_input:      Can be full path to file with ids, list of ids, or one id.
        :return:               (Dictionary of network outputs, dictionary of post-processed (by self.OutputGen) network outputs)
        """

        assert(self.model_handler is not None)  # Check if trainer.init() was called before.
        assert(hparams.synth_dir is not None)  # Directory to store the generated audio files has to be set.
        makedirs_safe(hparams.synth_dir)
        id_list = ModelTrainer._input_to_str_list(ids_input)

        self.logger.info("Start synthesising [{0}]".format(", ".join(str(i) for i in id_list)))
        t_start = timer()
        model_output, model_output_post = self._forward_batched(hparams, id_list, hparams.batch_size_synth, load_target=False, synth=True, benchmark=False, gen_figure=hparams.synth_gen_figure)
        t_training = timer() - t_start
        self.logger.info('Synthesis time for {} sample(s): {}'.format(len(id_list), timedelta(seconds=t_training)))

        return model_output, model_output_post

    def gen_figure(self, hparams, ids_input):
        """
        Generate figures for all given ids with the self.gen_figure_from_output function (has to be implemented).

        :param hparams:        Hyper-parameter container.
        :param ids_input:      Can be full path to file with ids, list of ids, or one id.
        :return:               (Dictionary of network outputs, dictionary of post-processed (by self.OutputGen) network outputs)
        """

        assert(self.model_handler is not None)  # Check if trainer.init() was called before.
        id_list = ModelTrainer._input_to_str_list(ids_input)

        self.logger.info("Start generating figures for [{0}]".format(", ".join(str(i) for i in id_list)))
        t_start = timer()
        model_output, model_output_post = self._forward_batched(hparams, id_list, hparams.batch_size_gen_figure, synth=False, benchmark=False, gen_figure=True)
        t_training = timer() - t_start
        self.logger.info('Figure generation time for {} sample(s): {}'.format(len(id_list), timedelta(seconds=t_training)))

        return model_output, model_output_post

    def benchmark(self, hparams, ids_input=None):
        """
        Benchmark the currently loaded model using the self.compute_score function (has to be implemented).

        :param hparams:        Hyper-parameter container.
        :param ids_input:      Can be full path to file with ids, list of ids, one id, or None.
                               If ids_inputs=None benchmark on test set if not None, otherwise on validation set.
        :return:               Score(s).
        """

        assert(callable(getattr(self, 'compute_score', None)))  # Function has to be implemented for this trainer.
        assert(self.model_handler is not None)  # Check if trainer.init() was called before.

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
            id_list = ModelTrainer._input_to_str_list(ids_input)
            self.logger.info("Start benchmark on given input ({}): [{}]"
                             .format(len(id_list), ", ".join(str(i) for i in id_list)))

        t_start = timer()
        model_scores = self._forward_batched(hparams,
                                             id_list,
                                             hparams.batch_size_benchmark,
                                             synth=False,
                                             benchmark=True,
                                             gen_figure=False)
        t_training = timer() - t_start
        self.logger.info('Benchmark time for {} sample(s): {}'.format(len(id_list), timedelta(seconds=t_training)))

        return model_scores

    def _forward_batched(self, hparams, id_list, batch_size, load_target=True, synth=False, benchmark=False, gen_figure=False):
        """
        Forward the features for the given ids in batches through the network.

        :param hparams:               Hyper-parameter container.
        :param id_list:               A list of ids for which the features are accessible by the self.InputGen object.
        :param batch_size:            Max size of a chunk of ids forwarded.
        :param load_target:           Give the target to the model when forwarded (used in teacher forcing).
        :param synth:                 Use the self.synthesize method to generate audio.
        :param benchmark:             Benchmark the given ids with the self.compute_score function.
        :param gen_figure:            Generate figures with the self.gen_figure_from_output function.
        :return:                      (Dictionary of outputs, dictionary of post-processed (by self.OutputGen) outputs)
        """

        self.logger.info("Get model outputs as batches of size {}.".format(min(batch_size, len(id_list))))
        dict_outputs = dict()
        dict_outputs_post = dict()
        dict_hiddens = dict()

        for batch_index in range(0, len(id_list), batch_size):
            batch_id_list = id_list[batch_index:batch_index+batch_size]

            inputs = list()
            for id_name in batch_id_list:
                # Load preprocessed sample and add it to the inputs with target value.
                inputs.append(self.dataset_train.getitem_by_name(id_name, load_target))  # No length check here.

            if self.batch_collate_fn is not None:
                batch_input_labels, batch_target_labels, seq_length_inputs, seq_length_output, *_, permutation = self.batch_collate_fn(inputs, common_divisor=hparams.num_gpus, batch_first=hparams.batch_first)
            else:
                batch_input_labels, batch_target_labels, seq_length_inputs, seq_length_output, *_, permutation = self.model_handler.prepare_batch(inputs, common_divisor=hparams.num_gpus, batch_first=hparams.batch_first)

            # Run forward pass of model.
            nn_output, nn_hidden = self.model_handler.forward(batch_input_labels, hparams, seq_length_inputs)

            # Retrieve output from batch.
            if self.batch_decollate_fn is not None:
                outputs, hiddens = self.batch_decollate_fn(nn_output, nn_hidden, seq_length_output, permutation, batch_first=hparams.batch_first)
            else:
                outputs, hiddens = self.split_batch(nn_output, nn_hidden, seq_length_output, permutation, batch_first=hparams.batch_first)

            # Post-process samples and generate a figure if requested.
            for idx, id_name in enumerate(batch_id_list):
                dict_outputs[id_name] = outputs[idx]
                dict_hiddens[id_name] = hiddens[idx] if hiddens is not None else None

                # If output is a list or tuple use only the first element for post-processing.
                if isinstance(outputs[idx], tuple) or isinstance(outputs[idx], list):
                    # Generate a figure if requested.
                    if gen_figure:
                        self.gen_figure_from_output(id_name, outputs[idx][0], hiddens[idx] if hiddens is not None else None, hparams)
                    dict_outputs_post[id_name] = self.dataset_train.postprocess_sample(outputs[idx][0])
                else:
                    # Generate a figure if requested.
                    if gen_figure:
                        self.gen_figure_from_output(id_name, outputs[idx], hiddens[idx] if hiddens is not None else None, hparams)
                    dict_outputs_post[id_name] = self.dataset_train.postprocess_sample(outputs[idx])

        if benchmark:
            # Implementation of compute_score is checked in benchmark function.
            return self.compute_score(dict_outputs_post, dict_hiddens, hparams)
        if synth:
            self.synthesize(id_list, dict_outputs_post, hparams)

        return dict_outputs, dict_outputs_post

    def gen_figure_from_output(self, id_name, output, hidden, hparams):
        raise NotImplementedError("Class {} doesn't implement gen_figure_from_output(id_name, output, hidden, hparams)"
                                  .format(self.__class__.__name__))

    def synth_ref(self, hparams, file_id_list):
        # Create reference audio files containing only the vocoder degradation.
        self.logger.info("Synthesise references for [{0}].".format(", ".join([id_name for id_name in file_id_list])))

        synth_dict = dict()
        for id_name in file_id_list:
            world_dir = hparams.world_dir if hasattr(hparams, "world_dir") and hparams.world_dir is not None\
                                          else os.path.join(self.OutputGen.dir_labels, self.dir_extracted_acoustic_features)
            # Load reference audio features.
            try:
                output = WorldFeatLabelGen.load_sample(id_name, world_dir, num_coded_sps=hparams.num_coded_sps)
            except FileNotFoundError as e1:
                try:
                    output = WorldFeatLabelGen.load_sample(id_name, world_dir, add_deltas=True, num_coded_sps=hparams.num_coded_sps)
                    coded_sp, lf0, vuv, bap = WorldFeatLabelGen.convert_to_world_features(output, True, hparams.num_coded_sps)
                    length = len(output)
                    lf0 = lf0.reshape(length, 1)
                    vuv = vuv.reshape(length, 1)
                    bap = bap.reshape(length, 1)
                    output = np.concatenate((coded_sp, lf0, vuv, bap), axis=1)
                except FileNotFoundError as e2:
                    self.logger.error("Cannot find extracted acoustic feature with or without deltas labels in {}.".format(world_dir))
                    raise Exception([e1, e2])
            synth_dict[id_name] = output

        # Add identifier to suffix.
        old_synth_file_suffix = hparams.synth_file_suffix
        hparams.synth_file_suffix = '_ref' + str(hparams.num_coded_sps) + 'sp'

        # Run the vocoder.
        ModelTrainer.synthesize(self, file_id_list, synth_dict, hparams)

        # Restore identifier.
        hparams.synth_file_suffix = old_synth_file_suffix

    def run_world_synth(self, synth_output, hparams):
        """Run the WORLD synthesize method."""
        fft_size = pyworld.get_cheaptrick_fft_size(hparams.synth_fs)

        save_dir = hparams.synth_dir if hparams.synth_dir is not None else hparams.out_dir if hparams.out_dir is not None else os.path.curdir
        for id_name, output in synth_output.items():
            logging.info("Synthesise {} with the WORLD vocoder.".format(id_name))

            coded_sp, lf0, vuv, bap = WorldFeatLabelGen.convert_to_world_features(output, contains_deltas=False, num_coded_sps=hparams.num_coded_sps)

            if hparams.do_post_filtering:
                coded_sp = merlin_post_filter(coded_sp, WorldFeatLabelGen.mgc_alpha)

            ln_sp = pysptk.mgc2sp(np.ascontiguousarray(coded_sp, dtype=np.float64), alpha=WorldFeatLabelGen.mgc_alpha, gamma=0.0, fftlen=fft_size)
            # sp = np.exp(sp.real * 2.0)
            # sp.imag = sp.imag * 180.0 / np.pi
            sp = np.exp(ln_sp.real)
            sp = np.power(sp.real / 32768.0, 2)
            # sp = np.power(sp.real / 32768.0, 2)
            # sp = np.exp(np.power(sp.real, 2))
            # sp = pyworld.decode_spectral_envelope(np.ascontiguousarray(coded_sp, np.float64), self.synth_fs, fft_size)  # Cepstral version.
            f0 = np.exp(lf0, dtype=np.float64)
            vuv[f0 < WorldFeatLabelGen.f0_silence_threshold] = 0  # WORLD throws an error for too small f0 values.
            f0[vuv == 0] = 0.0
            ap = pyworld.decode_aperiodicity(np.ascontiguousarray(bap.reshape(-1, 1), np.float64), hparams.synth_fs, fft_size)

            waveform = pyworld.synthesize(f0, sp, ap, hparams.synth_fs)
            waveform = waveform.astype(np.float32, copy=False)  # Does inplace conversion, if possible.

            # Always save as wav file first and convert afterwards if necessary.
            file_path = os.path.join(save_dir, "{}{}{}{}".format(os.path.basename(id_name),
                                                                 "_" + hparams.model_name if hparams.model_name is not None else "",
                                                                 hparams.synth_file_suffix, "_WORLD"))
            makedirs_safe(hparams.synth_dir)
            soundfile.write(file_path + ".wav", waveform, hparams.synth_fs)

            # Use PyDub for special audio formats.
            if hparams.synth_ext.lower() != 'wav':
                as_wave = pydub.AudioSegment.from_wav(file_path + ".wav")
                file = as_wave.export(file_path + "." + hparams.synth_ext, format=hparams.synth_ext)
                file.close()
                os.remove(file_path + ".wav")

    def run_r9y9wavenet_mulaw_world_feats_synth(self, synth_output, hparams):
        from idiaptts.src.neural_networks.pytorch.models.WaveNetWrapper import WaveNetWrapper
        from idiaptts.src.data_preparation.audio.RawWaveformLabelGen import RawWaveformLabelGen

        org_model_type = hparams.model_type
        hparams.model_type = WaveNetWrapper.IDENTIFIER

        synth_output = copy.copy(synth_output)

        input_fs_Hz = 1000.0 / hparams.frame_size_ms
        in_to_out_multiplier = hparams.frame_rate_output_Hz / input_fs_Hz
        # dir_world_features = os.path.join(self.OutputGen.dir_labels, self.dir_extracted_acoustic_features)
        input_gen = WorldFeatLabelGen(None,
                                      add_deltas=False,
                                      sampling_fn=partial(sample_linearly, in_to_out_multiplier=in_to_out_multiplier, dtype=np.float32))
        # Load normalisation parameters for wavenet input.
        try:
            norm_params_path = os.path.splitext(hparams.synth_vocoder_path)[0] + "_norm_params.npy"
            input_gen.norm_params = np.load(norm_params_path).reshape(2, -1)
        except FileNotFoundError:
            self.logger.error("Cannot find normalisation parameters for WaveNet input at {}."
                              "Please save them there with numpy.save().".format(norm_params_path))
            raise

        wavenet_model_handler = ModelHandlerPyTorch()
        wavenet_model_handler.model, *_ = wavenet_model_handler.load_model(hparams.synth_vocoder_path,
                                                                           hparams,
                                                                           verbose=False)

        for id_name, output in synth_output.items():
            logging.info("Synthesise {} with {} vocoder.".format(id_name, hparams.synth_vocoder_path))

            if hparams.do_post_filtering:
                coded_sp, lf0, vuv, bap = input_gen.convert_to_world_features(output,
                                                                              contains_deltas=input_gen.add_deltas,
                                                                              num_coded_sps=hparams.num_coded_sps)
                coded_sp = merlin_post_filter(coded_sp, WorldFeatLabelGen.mgc_alpha)
                output = input_gen.convert_from_world_features(coded_sp, lf0, vuv, bap)

            output = input_gen.preprocess_sample(output)

            # output (T x C) --transpose--> (C x T) --unsqueeze(0)--> (B x C x T).
            output = output.transpose()[None, ...]
            # Wavenet input has to be (B x C x T).
            output, _ = wavenet_model_handler.forward(output, hparams, batch_seq_lengths=(output.shape[-1],))
            output = output[0].transpose()  # Remove batch dim and transpose back to (T x C).
            # Revert mu-law quantization.
            output = output.argmax(axis=1)
            synth_output[id_name] = RawWaveformLabelGen.mu_law_companding_reversed(output, hparams.mu)

        if hasattr(hparams, 'bit_depth'):
            org_bit_depth = hparams.bit_depth
            hparams.bit_depth = 16
        else:
            org_bit_depth = None
            hparams.add_hparam("bit_depth", 16)

        # Add identifier to suffix.
        old_synth_file_suffix = hparams.synth_file_suffix
        hparams.synth_file_suffix += '_' + hparams.synth_vocoder

        self.run_raw_synth(synth_output, hparams)

        # Restore identifier.
        hparams.model_type = org_model_type
        hparams.setattr_no_type_check("synth_file_suffix", old_synth_file_suffix)  # Can be None, thus no type check.
        hparams.setattr_no_type_check("bit_depth", org_bit_depth)  # Can be None, thus no type check.

    def run_raw_synth(self, synth_output, hparams):
        """Use Pydub to synthesis audio from raw data given in the synth_output dictionary."""

        for id_name, raw in synth_output.items():
            logging.info("Save {} from raw waveform.".format(id_name))

            # Load raw data into pydub AudioSegment.
            # raw /= raw.abs().max()
            raw *= math.pow(2, hparams.bit_depth) / 2  # Expand to pydub range.
            raw = raw.astype(np.int16)
            audio_seg = AudioSegment(
                # raw audio data (bytes)
                data=raw.tobytes(),
                # 2 byte (16 bit) samples
                sample_width=2,
                # Hz frame rate
                frame_rate=hparams.frame_rate_output_Hz,
                # mono
                channels=1
            )
            audio_seg.set_frame_rate(hparams.synth_fs)

            # Save the audio.
            wav_file_path = os.path.join(hparams.synth_dir, "".join((os.path.basename(id_name).rsplit('.', 1)[0], "_",
                                                                     hparams.model_name, hparams.synth_file_suffix, ".",
                                                                     hparams.synth_ext)))
            audio_seg.export(wav_file_path, format=hparams.synth_ext)

    def synthesize(self, file_id_list, synth_output, hparams):

        # Create speaker subdirectories if necessary.
        for id_name in file_id_list:
            path_split = os.path.split(id_name)
            if len(path_split) > 2:
                makedirs_safe(os.path.join(hparams.synth_dir, *path_split[:-1]))

        if hparams.synth_vocoder == "WORLD":
            self.run_world_synth(synth_output, hparams)
        # elif hparams.synth_vocoder == "STRAIGHT":  # Add further vocoders here.
        elif hparams.synth_vocoder == "r9y9wavenet_mulaw_16k_world_feats_English":

            # If no path is given, use pre-trained model.
            if not hasattr(hparams, "synth_vocoder_path") or hparams.synth_vocoder_path is None:
                parent_dirs = os.path.realpath(__file__).split(os.sep)
                dir_root = str.join(os.sep, parent_dirs[:parent_dirs.index("IdiapTTS") + 1])
                hparams.synth_vocoder_path = os.path.join(dir_root, "idiaptts", "misc", "pretrained", "r9y9wavenet_quantized_16k_world_feats_English.nn")

            # Default quantization is with mu=255.
            if not hasattr(hparams, "mu") or hparams.mu is None:
                hparams.add_hparam("mu", 255)

            if hasattr(hparams, 'frame_rate_output_Hz'):
                org_frame_rate_output_Hz = hparams.frame_rate_output_Hz
                hparams.frame_rate_output_Hz = 16000
            else:
                org_frame_rate_output_Hz = None
                hparams.add_hparam("frame_rate_output_Hz", 16000)

            self.run_r9y9wavenet_mulaw_world_feats_synth(synth_output, hparams)

            # TODO: Convert to requested frame rate. if org_frame_rate_output_Hz != 16000:
            hparams.setattr_no_type_check("frame_rate_output_Hz", org_frame_rate_output_Hz)  # Can be None.
