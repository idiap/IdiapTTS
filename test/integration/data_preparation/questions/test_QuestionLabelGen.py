#
# Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#


import unittest


import os
import shutil

import numpy as np

from idiaptts.misc.utils import makedirs_safe
from idiaptts.src.data_preparation.questions.QuestionLabelGen import QuestionLabelGen


class TestQuestionLabelGen(unittest.TestCase):

    num_questions = 425

    @classmethod
    def setUpClass(cls):
        cls.dir_questions = os.path.join("integration", "fixtures",
                                         "questions")
        cls.file_questions = os.path.join("integration", "fixtures",
                                          "questions-en-radio_dnn_400.hed")
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

        label_dict, *extracted_norm_params = QuestionLabelGen.gen_data(
            dir_in=self.dir_labels,
            file_questions=self.file_questions,
            dir_out=dir_out,
            id_list=self.id_list,
            return_dict=True)

        question_gen = QuestionLabelGen(dir_out, num_questions=409)
        norm_params = question_gen.get_normalisation_params(dir_out)
        self.assertTrue((extracted_norm_params[0] == norm_params[0]).all())
        self.assertTrue((extracted_norm_params[1] == norm_params[1]).all())

        test_label = label_dict[self.id_list[1]]

        test_label_pre = question_gen.preprocess_sample(test_label)
        self.assertTrue(np.isclose(
            test_label_pre, question_gen[self.id_list[1]]).all())

        test_label_post = question_gen.postprocess_sample(test_label_pre)
        self.assertTrue(np.isclose(test_label, test_label_post).all())

        shutil.rmtree(dir_out)

    def test_load(self):
        sample = QuestionLabelGen.load_sample(self.id_list[0],
                                              self.dir_questions,
                                              num_questions=409)
        self.assertEqual(409, sample.shape[1])
