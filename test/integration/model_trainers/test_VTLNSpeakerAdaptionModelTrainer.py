#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#


import unittest

import os
import shutil
import numpy
import warnings

from idiaptts.src.model_trainers.vtln.VTLNSpeakerAdaptionModelTrainer import VTLNSpeakerAdaptionModelTrainer
from idiaptts.src.neural_networks.pytorch.utils import equal_checkpoint


class TestVTLNSpeakerAdaptionModelTrainer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load test data
        cls.dir_world_features = os.path.join("integration", "fixtures", "WORLD")
        cls.dir_question_labels = os.path.join("integration", "fixtures", "questions")
        cls.id_list = cls._get_id_list()

    @classmethod
    def tearDownClass(cls):
        hparams = cls._get_hparams(cls())
        os.rmdir(hparams.out_dir)  # Remove class name directory, should be empty.

    def _get_hparams(self):
        hparams = VTLNSpeakerAdaptionModelTrainer.create_hparams()
        # General parameters
        hparams.num_questions = 409
        hparams.voice = "full"
        hparams.data_dir = os.path.realpath(os.path.join("integration", "fixtures", "database"))
        hparams.out_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), type(self).__name__)

        hparams.frame_size_ms = 5
        hparams.num_coded_sps = 20
        hparams.seed = 1

        # Training parameters.
        hparams.epochs = 3
        hparams.use_gpu = False
        hparams.model_type = "VTLN"
        hparams.model_name = "VTLN.nn"
        hparams.batch_size_train = 2
        hparams.batch_size_val = 50
        hparams.use_saved_learning_rate = True
        hparams.optimiser_args["lr"] = 0.001
        hparams.model_name = "test_model.nn"
        hparams.epochs_per_checkpoint = 2

        # hparams.pass_embs_to_pre_net = False
        hparams.num_speakers = 2
        hparams.f_get_emb_index = (lambda id_name, length: numpy.zeros((length, hparams.num_speakers)),)
        hparams.pre_net_model_type = "RNNDYN-1_RELU_32-1_FC_67"
        hparams.pre_net_model_name = "pre-net.nn"

        return hparams

    @staticmethod
    def _get_id_list():
        with open(os.path.join("integration", "fixtures", "database", "file_id_list.txt")) as f:
            id_list = f.readlines()
        # Trim entries in-place.
        id_list[:] = [s.strip(' \t\n\r') for s in id_list]
        return id_list

    def test_init_create(self):
        hparams = self._get_hparams()
        hparams.out_dir = os.path.join(hparams.out_dir, "test_init_create")  # Add function name to path.

        trainer = VTLNSpeakerAdaptionModelTrainer(self.dir_world_features, self.dir_question_labels, self.id_list, hparams.num_questions, hparams)
        trainer.init(hparams)

        shutil.rmtree(hparams.out_dir)

    def test_init_load_prenet(self):
        hparams = self._get_hparams()
        hparams.out_dir = os.path.join(hparams.out_dir, "test_init_load_prenet")  # Add function name to path.
        hparams.pre_net_model_path = os.path.join("integration", "fixtures", "test_model_in409_out67.nn")

        trainer = VTLNSpeakerAdaptionModelTrainer(self.dir_world_features,
                                                  self.dir_question_labels,
                                                  self.id_list,
                                                  hparams.num_questions,
                                                  hparams)
        trainer.init(hparams)

        shutil.rmtree(hparams.out_dir)

    def test_save_load_equality(self):
        hparams = self._get_hparams()
        hparams.out_dir = os.path.join(hparams.out_dir, "test_save_load_equality")  # Add function name to path.
        model_path = os.path.join(hparams.out_dir, "nn", "test_model.nn")

        # Create a new model and save it.
        total_epochs = 10
        trainer = VTLNSpeakerAdaptionModelTrainer(self.dir_world_features,
                                                  self.dir_question_labels,
                                                  self.id_list,
                                                  hparams.num_questions,
                                                  hparams)
        trainer.init(hparams)
        trainer.model_handler.save_checkpoint(model_path, total_epochs)

        # Create a new model and test load.
        hparams.load_from_checkpoint = True
        hparams.model_type = None
        trainer = VTLNSpeakerAdaptionModelTrainer(self.dir_world_features,
                                                  self.dir_question_labels,
                                                  self.id_list,
                                                  hparams.num_questions,
                                                  hparams)
        trainer.init(hparams)
        trainer.model_handler.load_checkpoint(model_path, hparams)
        model_copy_path = os.path.join(hparams.out_dir, "test_model_copy.nn")
        trainer.model_handler.save_checkpoint(model_copy_path, total_epochs)

        # self.assertTrue(filecmp.cmp(model_path, model_copy_path, False))  # This does not work.
        self.assertTrue(equal_checkpoint(model_path, model_copy_path), "Loaded and saved models are not the same.")

        shutil.rmtree(hparams.out_dir)

    def test_train(self):
        hparams = self._get_hparams()
        hparams.out_dir = os.path.join(hparams.out_dir, "test_train")  # Add function name to path.
        hparams.seed = 1234
        hparams.use_best_as_final_model = False

        trainer = VTLNSpeakerAdaptionModelTrainer(self.dir_world_features,
                                                  self.dir_question_labels,
                                                  self.id_list,
                                                  hparams.num_questions,
                                                  hparams)
        trainer.init(hparams)
        _, all_loss_train, _ = trainer.train(hparams)

        # Training loss decreases?
        self.assertLess(all_loss_train[-1], all_loss_train[1 if hparams.start_with_test else 0],
                        msg="Loss did not decrease over {} epochs".format(hparams.epochs))

        shutil.rmtree(hparams.out_dir)

    def test_train_double_embedding(self):
        hparams = self._get_hparams()
        hparams.out_dir = os.path.join(hparams.out_dir, "test_train")  # Add function name to path.
        hparams.seed = 1234
        hparams.use_best_as_final_model = False
        hparams.pre_net_model_type = "RNNDYN-2x{}_EMB_(-1)-3x64_EMB_(-1)-1_RELU_32-1_FC_67".format(hparams.num_speakers)
        hparams.pass_embs_to_pre_net = True
        hparams.f_get_emb_index = (lambda id_name, length: numpy.zeros((length, 1)),
                                   lambda id_name, length: numpy.zeros((length, 1)))

        trainer = VTLNSpeakerAdaptionModelTrainer(self.dir_world_features,
                                                  self.dir_question_labels,
                                                  self.id_list,
                                                  hparams.num_questions,
                                                  hparams)
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

        trainer = VTLNSpeakerAdaptionModelTrainer(self.dir_world_features,
                                                  self.dir_question_labels,
                                                  self.id_list,
                                                  hparams.num_questions,
                                                  hparams)
        trainer.init(hparams)
        scores = trainer.benchmark(hparams)

        numpy.testing.assert_almost_equal((9.401, 78.124, 0.609, 38.964), scores, 3, "Wrong benchmark score.")

        shutil.rmtree(hparams.out_dir)

    def test_gen_figure(self):
        num_test_files = 2

        hparams = self._get_hparams()
        hparams.out_dir = os.path.join(hparams.out_dir, "test_gen_figure")  # Add function name to path
        hparams.pre_net_model_path = os.path.join("integration", "fixtures", "test_model_in409_out67.nn")

        trainer = VTLNSpeakerAdaptionModelTrainer(self.dir_world_features,
                                                  self.dir_question_labels,
                                                  self.id_list,
                                                  hparams.num_questions,
                                                  hparams)
        trainer.init(hparams)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
            trainer.gen_figure(hparams, self.id_list[:num_test_files])
        # Check number of created files.
        found_files = list([name for name in os.listdir(hparams.out_dir)
                            if os.path.isfile(os.path.join(hparams.out_dir, name))
                            and name.endswith(hparams.model_name + ".VTLN" + hparams.gen_figure_ext)])
        self.assertEqual(len(self.id_list[:num_test_files]), len(found_files),
                         msg="Number of {} files in out_dir directory does not match.".format(hparams.gen_figure_ext))

        shutil.rmtree(hparams.out_dir)
