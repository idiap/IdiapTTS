#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#

"""
Handler module base class.
"""

# System imports.
import logging

# Third-party imports.

# Local source tree imports.


class ModelHandler(object):
    """
    Provides functionality to work with multiple network architectures. It allows to create, load and save a model,
    train and test it and load the data for it. This class creates a wrapper around the framework used to implement
    the models. This class should be implemented for each backend (PyTorch, TensorFlow, etc.).
    """
    logger = logging.getLogger(__name__)

    #########################
    # Default constructor
    #
    def __init__(self):

        # Data variables.
        self.dataloader_train = None
        self.dataloader_val = None

        # Object variables.
        self.model = None
        self.model_name = None

        # Training parameters.
        self.loss_function = None
        self.optimiser = None
        self.scheduler = None

    def create_model(self, hparams, dim_in, dim_out):
        """
        Create a new model.

        :param hparams:           Hyper-parameter container.
        :param dim_in:            Input dimension of data.
        :param dim_out:           Output dimension of data.
        :return:                  Nothing. The model is created at self.model.
        """
        raise NotImplementedError("Class %s doesn't implement create_model()" % self.__class__.__name__)

    def load_checkpoint(self, file_path, hparams, initial_lr):
        """
        Load a checkpoint, also transfers to GPU if hparams.use_gpu is True.

        :param file_path:         Full path to checkpoint.
        :param hparams:           Hyper-parameter container. Needs to have use_gpu.
        :param initial_lr:        Learning rate of first epoch. Necessary when continuing training by some optimisers.
        :return:                  Nothing. The model is loaded to self.model.
        """
        raise NotImplementedError("Class %s doesn't implement load_checkpoint()" % self.__class__.__name__)

    def save_checkpoint(self, file_path):
        """Save a CPU version of the model combined with optimiser and other parameters."""
        raise NotImplementedError("Class %s doesn't implement save_checkpoint()" % self.__class__.__name__)

    def forward(self, in_tensor, hparams, batch_seq_lengths=None):
        """Forward one example through the model.

        :param in_tensor:            Correct tensor type or numpy array.
        :param hparams:              An object with hyper parameters for the model and the experiment.
        :param batch_seq_lengths:    List of sequence lengths in the batch (optional).
        :return:                     Output of model (numpy array).
        """
        raise NotImplementedError("Class %s doesn't implement forward()" % self.__class__.__name__)

    def process_dataloader(self, dataloader, hparams, training=True):
        """
        Process data in dataloader by loading batches from it.
        Perform training if training parameter is True, otherwise test run is assumed.

        :param dataloader:        Dataloader of the train set.
        :param hparams:           Hyper-parameter container.
        :param training:          Enables gradients, dropout, batch norm, etc.
        :return:                  Nothing.
        """
        raise NotImplementedError("Class %s doesn't implement train()" % self.__class__.__name__)

    def run(self, hparams, loss_function=None):
        """Run train and test method for the number of times specified in hparams.epochs."""
        raise NotImplementedError("Class %s doesn't implement run()" % self.__class__.__name__)
