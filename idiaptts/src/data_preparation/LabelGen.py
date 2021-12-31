#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#
import logging
import os

import numpy as np

from idiaptts.misc.utils import makedirs_safe


class LabelGen(object):
    """
    Abstract class for all label generators.

    The following methods have to be implemented by subclasses:
    """

    def gen_data(self, dir_in, dir_out=None, file_id_list=None, id_list=None, return_dict=False):
        raise NotImplementedError("Class %s doesn't implement gen_data(dir_in, dir_out, file_id_list, id_list, return_dict)" % self.__class__.__name__)

    # def get_dir_labels(self): # Unused 23.5.18
    #     raise NotImplementedError("Class %s doesn't implement get_dir_labels()" % (self.__class__.__name__))

    def __getitem__(self, id_name):
        raise NotImplementedError("Class %s doesn't implement __getitem__(id_name)" % self.__class__.__name__)

    def preprocess_sample(self, sample, norm_params=None):
        raise NotImplementedError("Class %s doesn't implement preprocess_sample(sample, norm_params)" % self.__class__.__name__)

    def postprocess_sample(self, sample, norm_params=None):
        raise NotImplementedError("Class %s doesn't implement postprocess_sample(sample, norm_params)" % self.__class__.__name__)

    @staticmethod
    def trim_end_sample(sample, length, reverse=False):
        """
        Trim the end of the given sample by the given length. If reverse is True, the front of the sample is trimmed.
        This function is called after preprocess_sample.
        """
        raise NotImplementedError("Class %s doesn't implement trim_end_sample(sample, length, reverse=False)" % LabelGen.__class__.__name__)

    # TODO: Implement this class instead of the previous method.
    def trim_sample(self, sample, trim_front, trim_end):
        sample = self.trim_end_sample(sample, trim_end)
        sample = self.trim_end_sample(sample, trim_front, reverse=True)
        return sample

    def pad(self, sample, pad_width):
        return np.pad(sample, pad_width)

    def trim(self, sample, trim_width):
        trim_width = [slice(v[0], sample.shape[dim] - v[1]) if isinstance(v, tuple) else v for dim, v in enumerate(trim_width)]
        return sample[tuple(trim_width)]

    @staticmethod
    def load_sample(id_name, dir_out):
        raise NotImplementedError("Class %s doesn't implement load_sample(id_name, dir_out)" % LabelGen.__class__.__name__)

    @staticmethod
    def _save_to_npz(file_path: os.PathLike, features: np.ndarray,
                     feature_name: str) -> None:

        makedirs_safe(os.path.dirname(file_path))
        if not file_path.endswith(".npz"):
            file_path += ".npz"
        file_path_backup = file_path + "_bak"

        clean_backup_file = False

        if os.path.isfile(file_path):
            saved_features = dict(np.load(file_path))

            os.rename(file_path, file_path_backup)
            clean_backup_file = True

            if feature_name in saved_features:
                logging.info("Overriding {} in {}.".format(
                    feature_name, file_path))
            saved_features[feature_name] = features
        else:
            saved_features = {feature_name: features}

        try:
            np.savez(file_path, **saved_features)
        except:
            if os.path.isfile(file_path_backup):
                logging.error("Error when writing {}, restoring backup".format(
                    file_path))
                if os.path.isfile(file_path):
                    os.remove(file_path)
                os.rename(file_path_backup, file_path)
                clean_backup_file = False
            else:
                logging.error("Error when writing {}.".format(file_path))
            raise

        if clean_backup_file:
            os.remove(file_path_backup)
