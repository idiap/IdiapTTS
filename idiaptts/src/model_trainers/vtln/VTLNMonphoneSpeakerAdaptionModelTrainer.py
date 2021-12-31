#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#

# System imports.

# Third-party imports.

# Local source tree imports.
from idiaptts.src.model_trainers.vtln.VTLNSpeakerAdaptionModelTrainer import VTLNSpeakerAdaptionModelTrainer
from idiaptts.src.model_trainers.EncDecMonophoneModelTrainer import EncDecMonophoneModelTrainer


# TODO: Remove this class alltogether, or at least rename to APWEncDecTrainer
class VTLNMonophoneSpeakerAdaptionModelTrainer(VTLNSpeakerAdaptionModelTrainer):

    @staticmethod
    def create_hparams(hparams_string=None, verbose=False):
        return EncDecMonophoneModelTrainer.create_hparams(hparams_string,
                                                          verbose)
