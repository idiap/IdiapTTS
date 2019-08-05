#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#


import unittest

import os
import shutil

from idiaptts.src.model_trainers.DurationModelTrainer import DurationModelTrainer
from idiaptts.src.model_trainers.AcousticModelTrainer import AcousticModelTrainer
from idiaptts.src.TTSModel import TTSModel


class TestTTSModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load test ids.
        cls.id_list = cls._get_id_list()

    @classmethod
    def tearDownClass(cls):
        hparams = cls._get_duration_hparams(cls())
        os.rmdir(hparams.out_dir)  # Remove class name directory, should be empty.

    @staticmethod
    def _get_id_list():
        with open(os.path.join("integration", "fixtures", "database", "file_id_list.txt")) as f:
            id_list = f.readlines()
        # Trim entries in-place.
        id_list[:] = [s.strip(' \t\n\r') for s in id_list]
        return id_list

    def _get_duration_hparams(self):
        hparams = DurationModelTrainer.create_hparams()
        hparams.out_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), type(self).__name__)
        hparams.seed = 1234
        hparams.epochs = 0
        hparams.use_gpu = False
        hparams.model_type = "RNNDYN-1_RELU_32-1_FC_5"
        hparams.model_name = "DM.nn"

        return hparams

    def _get_duration_trainer(self, hparams):

        dir_durations = os.path.join("integration", "fixtures", "dur")
        if hparams.phoneme_label_type == "HTK full":
            dir_phoneme_labels = os.path.join("integration", "fixtures", "labels", "label_state_align")
        elif hparams.phoneme_label_type == "mono_no_align":
            dir_phoneme_labels = os.path.join("integration", "fixtures", "labels", "mono_no_align")
        else:
            raise NotImplementedError("Unknown label type {}.".format(hparams.phoneme_label_type))
        file_symbol_dict = os.path.join("integration", "fixtures", "labels", "mono_phone.list")

        return DurationModelTrainer(dir_phoneme_labels, dir_durations, self.id_list, file_symbol_dict, hparams)

    def _get_acoustic_hparams(self):
        hparams = AcousticModelTrainer.create_hparams()
        # General parameters
        hparams.num_questions = 409
        # hparams.data_dir = os.path.realpath(os.path.join("integration", "fixtures", "database"))
        hparams.out_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), type(self).__name__)

        hparams.sampling_frequency = 16000
        hparams.frame_size_ms = 5
        hparams.num_coded_sps = 20
        hparams.seed = 1234

        # Training parameters.
        hparams.epochs = 0
        hparams.use_gpu = False
        hparams.model_type = "RNNDYN-1_RELU_32-1_FC_67"
        hparams.model_name = "AM.nn"

        return hparams

    def _get_acoustic_trainer(self, hparams):
        dir_world_features = os.path.join("integration", "fixtures", "WORLD")
        dir_question_labels = os.path.join("integration", "fixtures", "questions")
        return AcousticModelTrainer(dir_world_features, dir_question_labels, self.id_list, hparams.num_questions, hparams)

    def test_run_DM_AM(self):
        # Create a duration model.
        hparams_duration = self._get_duration_hparams()
        hparams_duration.out_dir = os.path.join(hparams_duration.out_dir, "test_run_DM_AM")
        duration_trainer = self._get_duration_trainer(hparams_duration)
        duration_trainer.init(hparams_duration)

        # Create an acoustic model.
        hparams_acoustic = self._get_acoustic_hparams()
        hparams_acoustic.out_dir = hparams_duration.out_dir  # Use the same out directory.
        acoustic_trainer = self._get_acoustic_trainer(hparams_acoustic)
        acoustic_trainer.init(hparams_acoustic)

        # Run TTS synthesis.
        hparams = AcousticModelTrainer.create_hparams()
        hparams.data_dir = os.path.realpath(os.path.join("integration", "fixtures", "database"))
        hparams.work_dir = os.path.realpath(os.path.join("integration", "fixtures"))
        hparams.synth_dir = hparams_duration.out_dir
        hparams.use_gpu = False
        tools_dir = os.path.join(os.path.dirname(os.path.realpath(".")), "tools")
        hparams.front_end = os.path.join(tools_dir, "tts_frontend", "English", "makeLabels.sh")
        hparams.front_end_accent = "AM"
        hparams.festival_dir = os.path.join(tools_dir, "festival")
        hparams.duration_labels_dir = os.path.join(hparams.work_dir, "dur")
        hparams.duration_model = hparams_duration.model_path
        hparams.file_symbol_dict = os.path.join(hparams.work_dir, "labels", "mono_phone.list")
        hparams.min_phoneme_length = 10000
        hparams.num_phoneme_states = 5
        hparams.file_questions = os.path.join(hparams.work_dir, "questions-en-radio_dnn_400.hed")
        hparams.num_questions = 409
        hparams.question_labels_norm_file = os.path.join(hparams.work_dir, "questions", "min-max.bin")
        hparams.world_features_dir = os.path.join(hparams.work_dir, "WORLD")
        hparams.num_coded_sps = 20
        hparams.acoustic_model = hparams_acoustic.model_path

        test_utterances = ["This is a test.", "Hello World!"]
        TTSModel.run_DM_AM(hparams, test_utterances)

        # Check if files were created.
        found_files = list([name for name in os.listdir(hparams.synth_dir)
                            if os.path.isfile(os.path.join(hparams.synth_dir, name))
                            and name.endswith(hparams.synth_ext)])
        # Check number of created files.
        self.assertEqual(len(test_utterances), len(found_files),
                         msg="Number of {} files in synth_dir directory does not match.".format(hparams.synth_ext))

        shutil.rmtree(hparams_duration.out_dir)
