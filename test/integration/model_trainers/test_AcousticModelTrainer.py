#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#


import os
import shutil
import warnings

import numpy
import soundfile
import unittest

from idiaptts.src.model_trainers.AcousticModelTrainer import AcousticModelTrainer


class TestAcousticModelTrainer(unittest.TestCase):

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
        hparams = AcousticModelTrainer.create_hparams()
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
        hparams.model_type = "RNNDYN-1_RELU_32-1_FC_67"
        hparams.batch_size_train = 2
        hparams.batch_size_val = 50
        hparams.use_saved_learning_rate = True
        hparams.optimiser_args["lr"] = 0.001
        hparams.model_name = "test_model"
        hparams.epochs_per_checkpoint = 2
        hparams.world_dir = self.dir_world_features

        return hparams

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

        trainer = AcousticModelTrainer(**AcousticModelTrainer.legacy_support_init(
            self.dir_world_features, self.dir_question_labels, self.id_list, hparams.num_questions, hparams))
        trainer.init(hparams)

        shutil.rmtree(hparams.out_dir)

    def test_train(self):
        hparams = self._get_hparams()
        hparams.out_dir = os.path.join(hparams.out_dir, "test_train")  # Add function name to path.
        hparams.seed = 1234
        hparams.use_best_as_final_model = False

        trainer = AcousticModelTrainer(**AcousticModelTrainer.legacy_support_init(
            self.dir_world_features, self.dir_question_labels, self.id_list, hparams.num_questions, hparams))
        trainer.init(hparams)
        _, all_loss_train, _ = trainer.train(hparams)
        all_loss_train = all_loss_train["MSELoss_acoustic_features"]

        # Training loss decreases?
        self.assertLess(all_loss_train[-1], all_loss_train[1 if hparams.start_with_test else 0],
                        msg="Loss did not decrease over {} epochs".format(hparams.epochs))

        shutil.rmtree(hparams.out_dir)

    def test_benchmark(self):
        hparams = self._get_hparams()
        hparams.out_dir = os.path.join(hparams.out_dir, "test_benchmark")  # Add function name to path.
        hparams.seed = 1

        trainer = AcousticModelTrainer(**AcousticModelTrainer.legacy_support_init(
            self.dir_world_features, self.dir_question_labels, self.id_list, hparams.num_questions, hparams))
        trainer.init(hparams)
        scores = trainer.benchmark(hparams)

        numpy.testing.assert_almost_equal((8.616, 78.4, 0.609, 37.352), scores["pred_acoustic_features"], 3, "Wrong benchmark score.")

        shutil.rmtree(hparams.out_dir)

    def test_gen_figure(self):
        num_test_files = 2

        hparams = self._get_hparams()
        hparams.out_dir = os.path.join(hparams.out_dir, "test_gen_figure")  # Add function name to path
        hparams.model_name = "test_model_in409_out67"
        hparams.model_path = os.path.join("integration", "fixtures", hparams.model_name, hparams.networks_dir)
        hparams.load_checkpoint_epoch = -1

        trainer = AcousticModelTrainer(**AcousticModelTrainer.legacy_support_init(
            self.dir_world_features, self.dir_question_labels, self.id_list, hparams.num_questions, hparams))
        trainer.init(hparams)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
            trainer.gen_figure(hparams, self.id_list[:num_test_files])
        # Check number of created files.
        found_files = list([name for name in os.listdir(os.path.join(hparams.out_dir, hparams.model_name, "figures"))
                            if os.path.isfile(os.path.join(hparams.out_dir, hparams.model_name, "figures", name))
                            and name.endswith(".Org-PyTorch" + hparams.gen_figure_ext)])
        self.assertEqual(len(self.id_list[:num_test_files]), len(found_files),
                         msg="Number of {} files in out_dir directory does not match.".format(hparams.gen_figure_ext))

        shutil.rmtree(hparams.out_dir)

    def test_synth_wav(self):
        num_test_files = 2

        hparams = self._get_hparams()
        hparams.out_dir = os.path.join(hparams.out_dir, "test_synth_wav")  # Add function name to path
        hparams.model_name = "test_model_in409_out67"
        hparams.model_path = os.path.join("integration", "fixtures", hparams.model_name, hparams.networks_dir)
        hparams.synth_fs = 16000
        hparams.frame_size_ms = 5
        hparams.synth_ext = "wav"
        hparams.synth_load_org_sp = True
        hparams.synth_load_org_lf0 = True
        hparams.synth_load_org_vuv = True
        hparams.synth_load_org_bap = True

        trainer = AcousticModelTrainer(**AcousticModelTrainer.legacy_support_init(
            self.dir_world_features, self.dir_question_labels, self.id_list, hparams.num_questions, hparams))
        trainer.init(hparams)
        hparams.synth_dir = os.path.join(hparams.out_dir, hparams.model_name)
        trainer.synth(hparams, self.id_list[:num_test_files])

        found_files = list([name for name in os.listdir(hparams.synth_dir)
                            if os.path.isfile(os.path.join(hparams.synth_dir, name))
                            and name.endswith("_WORLD." + hparams.synth_ext)])
        # Check number of created files.
        self.assertEqual(len(self.id_list[:num_test_files]), len(found_files),
                         msg="Number of {} files in synth_dir directory does not match.".format(hparams.synth_ext))

        # Check readability and length of one created file.
        raw, fs = soundfile.read(os.path.join(hparams.synth_dir, found_files[0]))
        self.assertEqual(hparams.synth_fs, fs, msg="Desired sampling frequency of output doesn't match.")
        labels = trainer.datareaders["acoustic_features"][[id_name for id_name in self.id_list[:num_test_files]
                                                           if id_name in found_files[0]][0]]
        expected_length = len(raw) / hparams.synth_fs / hparams.frame_size_ms * 1000
        self.assertTrue(abs(expected_length - len(labels["acoustic_features"])) < 10,
                        msg="Saved raw audio file length does not roughly match length of labels.")

        shutil.rmtree(hparams.out_dir)
