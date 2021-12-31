#
# Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#


import unittest

import os
import shutil

from idiaptts.misc.utils import makedirs_safe
from idiaptts.src.data_preparation.NpzDataReader import NpzDataReader
from idiaptts.src.data_preparation.phonemes.PhonemeDurationLabelGen import PhonemeDurationLabelGen


class TestPhonemeDurationLabelGen(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dir_labels_state = os.path.join("integration", "fixtures",
                                            "labels", "label_state_align")
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

        label_dict, *extracted_norm_params = PhonemeDurationLabelGen.gen_data(
            self.dir_labels_state, dir_out, id_list=self.id_list,
            label_type="full_state_align", return_dict=True)

        dur_gen = PhonemeDurationLabelGen(dir_labels=dir_out)
        norm_params = dur_gen.get_normalisation_params(dir_out)
        self.assertTrue((extracted_norm_params[0] == norm_params[0]).all())
        self.assertTrue((extracted_norm_params[1] == norm_params[1]).all())

        test_label = label_dict[self.id_list[1]]

        test_label_pre = dur_gen.preprocess_sample(test_label)
        self.assertTrue((test_label_pre == dur_gen[self.id_list[1]]).all())

        test_label_post = dur_gen.postprocess_sample(test_label_pre)
        self.assertTrue((test_label == test_label_post).all())

        self.assertRaises(AssertionError, PhonemeDurationLabelGen.Config,
                          name="durations", directory=dir_out,
                          load_as_matrix=True)
        self.assertRaises(AssertionError, PhonemeDurationLabelGen.Config,
                          name="durations", directory=dir_out,
                          pad_mode="edge", load_as_matrix=True)

        dur_gen = PhonemeDurationLabelGen(
            dir_labels=dir_out, pad_mode='edge',
            norm_type=NpzDataReader.Config.NormType.NONE, load_as_matrix=True)
        norm_params = dur_gen.get_normalisation_params(dir_out)

        self.assertIsNone(
            dur_gen.norm_params,
            "No normalisation should be used on attention matrix.")

        test_label = dur_gen[self.id_list[1]]

        expected_dims = (int(test_label_post.sum()), len(test_label_post))
        self.assertEqual(expected_dims, test_label.shape)

        shutil.rmtree(dir_out)
