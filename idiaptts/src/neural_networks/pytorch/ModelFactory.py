#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#

"""
Factory module to create PyTorch models from a model type string.
"""

# System imports.
import importlib
import logging

# Third-party imports.

# Local source tree imports.
from idiaptts.src.neural_networks.pytorch.models.RNNDyn import *
from idiaptts.src.neural_networks.pytorch.models.WarpingLayer import WarpingLayer


class ModelFactory(object):

    logger = logging.getLogger(__name__)

    @staticmethod
    def __register_architecture(registered_architectures, class_object):
        """Private class to allow class body initialisation with logging."""
        if class_object not in registered_architectures:
            logging.getLogger("ModelFactory").debug("Register architecture {}.".format(class_object.IDENTIFIER))
            registered_architectures.append(class_object)
        else:
            raise ValueError("{} architecture already registered.".format(class_object.IDENTIFIER))

    @staticmethod
    def register_architecture(class_object):
        """Adds the class object to the list registered_architectures."""
        ModelFactory.__register_architecture(ModelFactory.registered_architectures, class_object)

    @staticmethod
    def deregister_architecture(identifier):
        new_arch_list = []
        removed_arch = 0
        for arch in ModelFactory.registered_architectures:
            if arch.IDENTIFIER != identifier:
                new_arch_list.append(arch)
            else:
                removed_arch += 1

        if removed_arch == 0:
            raise KeyError("Architecture with identifier {} is unknown, thus cannot be unregistered.")
        if removed_arch > 1:
            ModelFactory.logger.warning("{} architectures were unregistered.")

        ModelFactory.logger.debug("Deregister {}.".format(identifier))
        ModelFactory.registered_architectures = new_arch_list

    # List of registered architectures.
    registered_architectures = list()

    # Register different architectures.
    __register_architecture.__func__(registered_architectures, RNNDyn)
    __register_architecture.__func__(registered_architectures, MerlinAcoustic)
    __register_architecture.__func__(registered_architectures, Interspeech18baseline)
    __register_architecture.__func__(registered_architectures, BaselineRNN_Yamagishi)
    __register_architecture.__func__(registered_architectures, Icassp19baseline)
    __register_architecture.__func__(registered_architectures, WarpingLayer)

    # Register optional architectures.
    requirement_neuralfilters = importlib.util.find_spec("neural_filters")
    if requirement_neuralfilters:
        from idiaptts.src.neural_networks.pytorch.models.NeuralFilters import NeuralFilters
        from idiaptts.src.neural_networks.pytorch.models.PhraseNeuralFilters import PhraseNeuralFilters
        __register_architecture.__func__(registered_architectures, PhraseNeuralFilters)
        __register_architecture.__func__(registered_architectures, NeuralFilters)

    requirement_wavenet_vocoder = importlib.util.find_spec("wavenet_vocoder")
    if requirement_wavenet_vocoder:
        from idiaptts.src.neural_networks.pytorch.models.WaveNetWrapper import WaveNetWrapper
        __register_architecture.__func__(registered_architectures, WaveNetWrapper)

    # requirement_nvtacotron2 = importlib.util.find_spec("tools.tacotron2")
    # if requirement_nvtacotron2:
    #     from idiaptts.src.neural_networks.pytorch.models.NVTacotron2Wrapper import NVTacotron2Wrapper
    #     __register_architecture.__func__(registered_architectures, NVTacotron2Wrapper)

    @staticmethod
    def _type_to_class(model_type):
        """
        Search in the registered architectures to fine a matching identifier.

        :param model_type:     Type of the model starting with the unique model identifier.
        :return:               A new model with PyTorch parameter initialization.
        """
        if type(model_type) is not str:
            raise TypeError("Expected string for model_type but received {}.".format(type(model_type)))

        for architecture in ModelFactory.registered_architectures:
            if re.match(architecture.IDENTIFIER, model_type) is not None:
                return architecture

        raise TypeError("Unknown network type: {}. No model was created.".format(model_type))

    @staticmethod
    def create(model_type, dim_in, dim_out, hparams, verbose=True):
        """Create a new instance of the class with PyTorch parameter initialization."""
        model_class = ModelFactory._type_to_class(model_type)

        user_defined_model_type = hparams.model_type
        hparams.model_type = model_type  # Some classes require a valid model_type, which could be None otherwise.
        model = model_class(dim_in, dim_out, hparams)
        hparams.model_type = user_defined_model_type

        # Send model to gpu, if requested.
        if hparams.use_gpu:
            if verbose:
                ModelFactory.logger.info("Convert network to GPU.")
            model = model.cuda()

        return model
