#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#


import unittest

import os
import shutil
import numpy
import warnings

from idiaptts.src.model_trainers.DurationModelTrainer import DurationModelTrainer


class TestDurationModelTrainer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load test ids.
        cls.id_list = cls._get_id_list()

    @classmethod
    def tearDownClass(cls):
        hparams = cls._get_hparams(cls())
        os.rmdir(hparams.out_dir)  # Remove class name directory, should be empty.

    def _get_hparams(self):
        hparams = DurationModelTrainer.create_hparams()
        # General parameters
        # hparams.voice = "full"
        hparams.data_dir = os.path.realpath(os.path.join("integration", "fixtures", "database"))
        hparams.out_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), type(self).__name__)

        hparams.seed = 1

        # Training parameters.
        hparams.epochs = 3
        hparams.use_gpu = False
        hparams.model_type = "RNNDYN-1_RELU_32-1_FC_5"
        hparams.batch_size_train = 64
        hparams.batch_size_val = 64
        hparams.learning_rate = 0.001
        hparams.model_name = "test_model.nn"
        hparams.epochs_per_checkpoint = 2

        return hparams

    def _get_trainer(self, hparams):

        dir_durations = os.path.join("integration", "fixtures", "dur")
        if hparams.phoneme_label_type == "HTK full":
            dir_phoneme_labels = os.path.join("integration", "fixtures", "labels", "label_state_align")
        elif hparams.phoneme_label_type == "mono_no_align":
            dir_phoneme_labels = os.path.join("integration", "fixtures", "labels", "mono_no_align")
        else:
            raise NotImplementedError("Unknown label type {}.".format(hparams.phoneme_label_type))
        file_symbol_dict = os.path.join("integration", "fixtures", "labels", "mono_phone.list")

        return DurationModelTrainer(dir_phoneme_labels, dir_durations, self.id_list, file_symbol_dict, hparams)

    @staticmethod
    def _get_id_list():
        with open(os.path.join("integration", "fixtures", "database", "file_id_list.txt")) as f:
            id_list = f.readlines()
        # Trim entries in-place.
        id_list[:] = [s.strip(' \t\n\r') for s in id_list]
        return id_list

    def test_init(self):
        hparams = self._get_hparams()
        hparams.out_dir = os.path.join(hparams.out_dir, "test_init")  # Add function name to path.

        trainer = self._get_trainer(hparams)
        trainer.init(hparams)

        shutil.rmtree(hparams.out_dir)

    def test_train(self):
        hparams = self._get_hparams()
        hparams.out_dir = os.path.join(hparams.out_dir, "test_train")  # Add function name to path.
        hparams.seed = 1234
        hparams.use_best_as_final_model = False

        trainer = self._get_trainer(hparams)
        trainer.init(hparams)
        _, all_loss_train, _ = trainer.train(hparams)

        # Training loss decreases?
        self.assertLess(all_loss_train[-1], all_loss_train[1 if hparams.start_with_test else 0],
                        msg="Loss did not decrease over {} epochs".format(hparams.epochs))

        shutil.rmtree(hparams.out_dir)

    def test_benchmark(self):
        hparams = self._get_hparams()
        hparams.out_dir = os.path.join(hparams.out_dir, "test_benchmark")  # Add function name to path.
        hparams.seed = 1

        trainer = self._get_trainer(hparams)
        trainer.init(hparams)
        scores = trainer.benchmark(hparams)

        numpy.testing.assert_almost_equal(286.609, scores, 3, "Wrong benchmark score.")

        shutil.rmtree(hparams.out_dir)

    def test_forward(self):
        hparams = self._get_hparams()
        hparams.out_dir = os.path.join(hparams.out_dir, "test_forward")  # Add function name to path.
        hparams.seed = 1234

        trainer = self._get_trainer(hparams)
        trainer.init(hparams)

        hparams.min_phoneme_length = 1000
        output, output_post = trainer.forward(hparams, self.id_list, only_positive=True)

        self.assertEqual(len(output), len(output_post),
                         msg="Length of output and post-processed output do not match length.")
        self.assertEqual(len(self.id_list), len(output_post), msg="Length of output and input do not match.")

        for key, out in output_post.items():
            self.assertEqual(0, (out % hparams.min_phoneme_length).max(),
                             msg="Post-processed output of {} is not a multiple of min_phoneme_length ({})."
                                 .format(key, hparams.min_phoneme_length))
            self.assertTrue((out >= 0).all(), msg="Post-processed output is not all positive.")

        shutil.rmtree(hparams.out_dir)
