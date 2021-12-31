#
# Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#

import unittest

import os

from idiaptts.misc.utils import makedirs_safe
from idiaptts.src.data_preparation.phonemes.PhonemeLabelGen import PhonemeLabelGen


class TestPhonemeLabelGen(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.id_list = cls._get_id_list()[:3]

        cls.dir_labels_state = os.path.join("integration", "fixtures",
                                            "labels", "label_state_align")
        cls.dir_labels_mono = os.path.join("integration", "fixtures",
                                           "labels", "mono_no_align")
        cls.dir_labels_full = os.path.join("integration", "fixtures",
                                           "labels", "full")
        cls.dir_labels_mfa = os.path.join("integration", "fixtures",
                                          "labels", "mfa")

        cls.file_symbol_dict = os.path.join("integration", "fixtures",
                                            "labels", "mono_phone.list")
        cls.file_symbol_dict_arpabet = os.path.join("integration", "fixtures",
                                                    "labels",
                                                    "phoneset_arpabet.txt")

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

    @staticmethod
    def _mask_even(sample):
        sample[::2] = 0
        return sample

    def test_legacy_load(self):
        label_types = ["full_state_align", "mono_no_align", "HTK full"]
        label_dirs = [self.dir_labels_state,
                      self.dir_labels_mono,
                      self.dir_labels_full]
        files_symbol_dict = [self.file_symbol_dict,
                             self.file_symbol_dict,
                             self.file_symbol_dict]
        exp_max_idx = [34, 34, 34]

        for label_type, label_dir, file_symbol_dict, exp_idx in zip(
                label_types, label_dirs, files_symbol_dict, exp_max_idx):
            label_gen = PhonemeLabelGen(label_dir,
                                        file_symbol_dict,
                                        label_type)
            test_label = label_gen[self.id_list[1]]

            self.assertEqual(1, test_label.shape[1])
            self.assertEqual(exp_idx, test_label.max())

            label_gen = PhonemeLabelGen(label_dir,
                                        file_symbol_dict,
                                        label_type,
                                        one_hot=True)

            test_label = label_gen[self.id_list[1]]

            symbol_dict = PhonemeLabelGen.get_symbol_dict(self.file_symbol_dict)
            self.assertEqual(len(symbol_dict), test_label.shape[1])
            self.assertTrue((test_label.max(axis=1) == 1).all())

            label_gen = PhonemeLabelGen(label_dir,
                                        file_symbol_dict,
                                        label_type,
                                        one_hot=True,
                                        add_EOF=True)

            test_label_with_eof = label_gen[self.id_list[1]]

            self.assertEqual(len(test_label) + 1, len(test_label_with_eof))
            self.assertEqual(1, test_label_with_eof[-1][-1])

    def test_load(self):

        label_types = ["full_state_align", "mono_no_align", "HTK full", "mfa"]
        label_dirs = [self.dir_labels_state,
                      self.dir_labels_mono,
                      self.dir_labels_full,
                      self.dir_labels_mfa]
        files_symbol_dict = [self.file_symbol_dict,
                             self.file_symbol_dict,
                             self.file_symbol_dict,
                             self.file_symbol_dict_arpabet]
        exp_max_idx = [34, 34, 34, 73]

        for label_type, label_dir, file_symbol_dict, exp_idx in zip(
                label_types, label_dirs, files_symbol_dict, exp_max_idx):

            label_gen = PhonemeLabelGen.Config(
                name="phonemes",
                directory=label_dir,
                file_symbol_dict=file_symbol_dict,
                label_type=label_type
            ).create_reader()

            test_label = label_gen[self.id_list[1]]["phonemes"]

            self.assertEqual(1, test_label.shape[1])
            self.assertEqual(exp_idx, test_label.max())

            label_gen = PhonemeLabelGen.Config(
                name="phonemes",
                directory=label_dir,
                file_symbol_dict=file_symbol_dict,
                label_type=label_type,
                one_hot=True
            ).create_reader()

            test_label = label_gen[self.id_list[1]]["phonemes"]

            symbol_dict = PhonemeLabelGen.get_symbol_dict(file_symbol_dict)
            self.assertEqual(len(symbol_dict), test_label.shape[1])
            self.assertTrue((test_label.max(axis=1) == 1).all())

            label_gen = PhonemeLabelGen.Config(
                name="phonemes",
                directory=label_dir,
                file_symbol_dict=file_symbol_dict,
                label_type=label_type,
                one_hot=True,
                add_EOF=True
            ).create_reader()

            test_label_with_eof = label_gen[self.id_list[1]]["phonemes"]

            self.assertEqual(len(test_label) + 1, len(test_label_with_eof))
            self.assertEqual(1, test_label_with_eof[-1][-1])

            label_gen = PhonemeLabelGen.Config(
                name="phonemes",
                directory=label_dir,
                file_symbol_dict=file_symbol_dict,
                label_type=label_type,
                one_hot=False,
                add_EOF=True,
                preprocessing_fn=self._mask_even
            ).create_reader()

            test_label_masked = label_gen[self.id_list[1]]["phonemes"]
            self.assertEqual(0, test_label_masked[2])
