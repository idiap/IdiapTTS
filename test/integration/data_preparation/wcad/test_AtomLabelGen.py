#
# Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#


import unittest


import os
import shutil

import numpy as np

from idiaptts.misc.utils import makedirs_safe
from idiaptts.src.data_preparation.wcad.AtomLabelGen import AtomLabelGen


class TestAtomLabelGen(unittest.TestCase):

    num_questions = 425

    @classmethod
    def setUpClass(cls):
        cls.dir_database = os.path.realpath(os.path.join(
            "integration", "fixtures", "database"))
        cls.dir_wav = os.path.join(cls.dir_database, "wav")
        cls.dir_atoms = os.path.join("integration", "fixtures",
                                         "wcad-0.030_0.060_0.090_0.120_0.150")
        cls.dir_wcad_root = os.path.join("IdiapTTS", "tools", "wcad")
        cls.id_list = cls._get_id_list()[:3]

    @staticmethod
    def _get_id_list():
        with open(os.path.join("integration", "fixtures", "database",
                               "file_id_list.txt")) as f:
            id_list = f.readlines()
        id_list[:] = [s.strip(' \t\n\r') for s in id_list]
        return id_list

    def _get_test_dir(self):
        out_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                               type(self).__name__)
        makedirs_safe(out_dir)
        return out_dir

    def test_save_load(self):
        dir_out = self._get_test_dir()

        theta_start = 0.01
        theta_stop = 0.055
        theta_step = 0.005
        thetas = np.arange(theta_start, theta_stop, theta_step)
        k = 6
        frame_size_ms = 5

        atom_gen = AtomLabelGen(self.dir_wcad_root, dir_out, thetas, k,
                                frame_size_ms)
        label_dict, *extracted_norm_params = atom_gen.gen_data(
            self.dir_wav, dir_out, id_list=self.id_list, return_dict=True)

        # Call this once before starting the pre-processing.
        norm_params = atom_gen.get_normalisation_params(dir_out)
        self.assertTrue((extracted_norm_params[0] == norm_params[0]).all())
        self.assertTrue((extracted_norm_params[1] == norm_params[1]).all())

        test_label = label_dict[self.id_list[1]]
        test_label_pre = atom_gen.preprocess_sample(test_label)
        self.assertTrue(np.isclose(test_label_pre,
                                   atom_gen[self.id_list[1]]).all())

        test_label_post = atom_gen.postprocess_sample(test_label_pre)
        # Post-precessing does peak selection, so pre and post labels
        # are not the same and we cannot check for equality here.

        self.assertTrue(np.isclose(-0.2547, test_label_post.sum(),
                                   atol=0.0001))

        os.remove(os.path.join(self.dir_database, "wcad_.txt"))
        shutil.rmtree(dir_out)

    def test_load(self):
        sample = AtomLabelGen.load_sample(self.id_list[0], self.dir_atoms,
                                          num_thetas=5)
        self.assertEqual(1931, sample.shape[0])
