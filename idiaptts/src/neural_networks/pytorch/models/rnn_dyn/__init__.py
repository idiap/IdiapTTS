#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#
IDENTIFIER = "RNNDYN"

from .Config import Config
from .RNNDyn import RNNDyn


def convert_legacy_to_config(in_dim, hparams):
    return RNNDyn._get_config_from_legacy_string(in_dim, None, hparams)
