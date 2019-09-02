#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#

# System imports.
import logging
import types

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
