#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#


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
    # @staticmethod
    # def trim_sample(sample, length_front, length_end):

    @staticmethod
    def load_sample(id_name, dir_out):
        raise NotImplementedError("Class %s doesn't implement load_sample(id_name, dir_out)" % LabelGen.__class__.__name__)
