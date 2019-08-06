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


class ModelFactory(object):

    logger = logging.getLogger(__name__)

    def __init__(self):
        # List of registered architectures.
        self.registered_architectures = list()

        # Register different architectures.
        self.register_architecture(RNNDyn)
        self.register_architecture(MerlinAcoustic)
        self.register_architecture(Interspeech18baseline)
        self.register_architecture(BaselineRNN_Yamagishi)
        self.register_architecture(Icassp19baseline)

        # Register optional architectures.
        requirement_warping_layer = importlib.util.find_spec("WarpingLayer")
        if requirement_warping_layer:
            from idiaptts.src.neural_networks.pytorch.models.WarpingLayer import WarpingLayer
            self.register_architecture(WarpingLayer)

        requirement_neuralfilters = importlib.util.find_spec("neural_filters")
        if requirement_neuralfilters:
            from idiaptts.src.neural_networks.pytorch.models.NeuralFilters import NeuralFilters
            from idiaptts.src.neural_networks.pytorch.models.PhraseNeuralFilters import PhraseNeuralFilters
            self.register_architecture(PhraseNeuralFilters)
            self.register_architecture(NeuralFilters)

        requirement_wavenet_vocoder = importlib.util.find_spec("wavenet_vocoder")
        if requirement_wavenet_vocoder:
            from idiaptts.src.neural_networks.pytorch.models.WaveNetWrapper import WaveNetWrapper
            self.register_architecture(WaveNetWrapper)

        requirement_nvtacotron2 = importlib.util.find_spec("tools.tacotron2")
        if requirement_nvtacotron2:
            from idiaptts.src.neural_networks.pytorch.models.NVTacotron2Wrapper import NVTacotron2Wrapper
            self.register_architecture(NVTacotron2Wrapper)

    def register_architecture(self, class_object):
        """Adds the class object to the list registered_architectures."""
        self.registered_architectures.append(class_object)

    def _type_to_class(self, model_type, dim_in, dim_out, hparams):
        """
        Search in the registered architectures if their identifier is matched.

        :param model_type:     Type of the model starting with the unique model identifier.
        :return:               A new model with PyTorch parameter initialization.
        """
        if type(model_type) is not str:
            raise TypeError("Expected string for model_type but received {}.".format(type(model_type)))

        for architecture in self.registered_architectures:
            if re.match(architecture.IDENTIFIER, model_type) is not None:
                return architecture(dim_in, dim_out, hparams)

        raise TypeError("Unknown network type: {}. No model was created.".format(model_type))

    def create(self, model_type, dim_in, dim_out, hparams, verbose=True):
        """Create a new instance of the class with PyTorch parameter initialization."""
        model = self._type_to_class(model_type, dim_in, dim_out, hparams)

        # Send model to gpu, if requested.
        if hparams.use_gpu:
            if verbose:
                self.logger.info("Convert network to GPU.")
            model = model.cuda()

        return model
