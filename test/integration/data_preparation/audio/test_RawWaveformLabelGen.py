#
# Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#


import unittest


import os
import shutil

import numpy as np

from idiaptts.misc.utils import makedirs_safe
from idiaptts.src.data_preparation.audio.RawWaveformLabelGen import RawWaveformLabelGen


class TestRawWaveformLabelGen(unittest.TestCase):

    output_frequency_Hz = 22050
    mu = 128

    @classmethod
    def setUpClass(cls):
        cls.dir_wav = os.path.join("integration", "fixtures", "database", "wav")
        cls.dir_labels = os.path.join("integration", "fixtures", "labels",
                                      "label_state_align")
        cls.id_list = cls._get_id_list()[:3]

    @staticmethod
    def _get_id_list():
        with open(os.path.join("integration", "fixtures", "database",
                               "file_id_list.txt")) as f:
            id_list = f.readlines()
        # Trim entries in-place.
        id_list[:] = [s.strip(' \t\n\r') for s in id_list]
        return id_list

    def _get_test_dir(self):
        out_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                               type(self).__name__)
        makedirs_safe(out_dir)
        return out_dir

    def test_save_load(self):
        dir_out = self._get_test_dir()

        raw_feat_gen = RawWaveformLabelGen(
            frame_rate_output_Hz=self.output_frequency_Hz, mu=self.mu)

        id_list = [os.path.join(self.dir_wav, id_ + ".wav")
                   for id_ in self.id_list]

        test_label = raw_feat_gen.load_sample(id_list[1],
                                              self.output_frequency_Hz)

        test_label_pre = raw_feat_gen.preprocess_sample(test_label)
        self.assertTrue(np.isclose(test_label_pre,
                                   raw_feat_gen[id_list[1]]).all())

        test_label_post = raw_feat_gen.postprocess_sample(test_label_pre)
        # Slightly different because quantisation loses information.
        self.assertLess(abs(test_label - test_label_post).max(), 0.0334)

        shutil.rmtree(dir_out)
