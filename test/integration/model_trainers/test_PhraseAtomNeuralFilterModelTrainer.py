# #
# # Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# # Written by Bastian Schnell <bastian.schnell@idiap.ch>
# #


# import unittest

# import os
# import shutil
# import numpy
# import soundfile
# import copy
# import itertools
# import logging
# import warnings

# from idiaptts.src.model_trainers.wcad.PhraseAtomNeuralFilterModelTrainer import PhraseAtomNeuralFilterModelTrainer


# class TestPhraseAtomNeuralFilterModelTrainer(unittest.TestCase):

#     @classmethod
#     def setUpClass(cls):
#         # Load test data
#         cls.id_list = cls._get_id_list()

#     @classmethod
#     def tearDownClass(cls):
#         hparams = cls._get_hparams(cls())
#         os.rmdir(hparams.out_dir)  # Remove class name directory, should be empty.

#     def _get_hparams(self, sub_dir=""):
#         hparams = PhraseAtomNeuralFilterModelTrainer.create_hparams()
#         # General parameters
#         hparams.num_questions = 409
#         theta_start = 0.03
#         theta_stop = 0.164
#         theta_step = 0.030
#         hparams.thetas = numpy.arange(theta_start, theta_stop, theta_step)
#         hparams.k = 2
#         hparams.min_atom_amp = 0.25
#         hparams.dist_window_size = 31  # [frames] should be odd.
#         hparams.data_dir = os.path.realpath(os.path.join("integration", "fixtures", "database"))
#         hparams.out_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), type(self).__name__, sub_dir)
#         hparams.save_final_model = True

#         hparams.frame_size_ms = 5
#         hparams.num_coded_sps = 20
#         hparams.seed = 1
#         hparams.complex_poles = False
#         hparams.phrase_bias_init = 4.5

#         # Training parameters.
#         hparams.epochs = 3
#         hparams.use_gpu = False
#         hparams.model_type = "PhraseNeuralFilters"
#         hparams.batch_size_train = 2
#         hparams.batch_size_val = 2
#         hparams.use_saved_learning_rate = True
#         hparams.optimiser_args["lr"] = 0.0006
#         hparams.model_path = os.path.join("integration", "fixtures", "phrase_neural_filters_model_in409_out2.nn")
#         hparams.model_name = "test_model.nn"
#         hparams.epochs_per_checkpoint = 2
#         hparams.use_best_as_final_model = False

#         hparams.vuv_loss_weight = 0.1
#         hparams.L1_loss_weight = 0.1
#         hparams.weight_unvoiced = 0

#         hparams_flat = copy.deepcopy(hparams)
#         # Training parameters.
#         hparams_flat.epochs = 0
#         hparams_flat.load_from_checkpoint = True
#         hparams_flat.model_type = "NeuralFilters"
#         # hparams_flat.model_type = None
#         hparams_flat.model_name = hparams.model_name + "_flat"
#         hparams_flat.model_path = os.path.join("integration", "fixtures", "neural_filters_model_in409_out2.nn")
#         hparams_flat.batch_size_train = 5
#         hparams_flat.optimiser_args["lr"] = 0.001
#         hparams_flat.complex_poles = False

#         hparams_atom = copy.deepcopy(hparams)
#         hparams_atom.synth_gen_figure = False
#         hparams_atom.model_type = "RNNDYN-1_RELU_32-1_FC_7"
#         hparams_atom.model_name = hparams_flat.model_name + "_atoms"
#         hparams_atom.model_path = os.path.join("integration", "fixtures", "test_model_in409_out7.nn")
#         hparams_atom.optimiser_args["lr"] = 0.0002
#         hparams_atom.batch_size_train = 2
#         hparams_atom.epochs = 0
#         hparams_atom.load_from_checkpoint = True

#         # Register hyper-parameter containers of subtrainers.
#         hparams.hparams_atom = hparams_atom
#         hparams.hparams_flat = hparams_flat
#         hparams_flat.hparams_atom = hparams_atom
#         # Set path to pre-trained models in fixtures directory.
#         hparams.flat_model_path = os.path.join("integration", "fixtures", "neural_filters_model_in409_out2.nn")
#         hparams.atom_model_path = os.path.join("integration", "fixtures", "test_model_in409_out7.nn")
#         hparams_flat.atom_model_path = hparams.atom_model_path

#         return hparams

#     @staticmethod
#     def _get_id_list():
#         with open(os.path.join("integration/fixtures/database/file_id_list.txt")) as f:
#             id_list = f.readlines()
#         # Trim entries in-place.
#         id_list[:] = [s.strip(' \t\n\r') for s in id_list]
#         return id_list

#     def _get_trainer(self, hparams):
#         dir_wcad_root = "../tools/wcad"
#         dir_audio = os.path.join(hparams.data_dir, "wav")
#         dir_world_features = "integration/fixtures/WORLD"
#         dir_atom_features = os.path.join("integration", "fixtures", "wcad-" + "_".join(map("{:.3f}".format, hparams.hparams_flat.thetas)))
#         dir_question_labels = "integration/fixtures/questions"

#         with warnings.catch_warnings():
#             warnings.filterwarnings("ignore", category=UserWarning, module="torch")
#             trainer = PhraseAtomNeuralFilterModelTrainer(dir_wcad_root,
#                                                          dir_audio,
#                                                          dir_atom_features,
#                                                          dir_world_features,
#                                                          dir_question_labels,
#                                                          self.id_list,
#                                                          hparams.hparams_flat.thetas,
#                                                          hparams.hparams_flat.k,
#                                                          hparams.num_questions,
#                                                          hparams.dist_window_size,
#                                                          hparams)

#         return trainer

#     def test_init(self):
#         hparams = self._get_hparams("test_init")

#         trainer = self._get_trainer(hparams)
#         trainer.init_atom(hparams)
#         trainer.init_flat(hparams)
#         trainer.init(hparams)

#         shutil.rmtree(hparams.out_dir)

#     def test_train(self):
#         hparams = self._get_hparams("test_train")
#         hparams.seed = 1234
#         trainer = self._get_trainer(hparams)

#         trainer.init_atom(hparams)
#         self.assertFalse(os.path.isfile(os.path.join(hparams.out_dir, hparams.networks_dir, hparams.hparams_flat.hparams_atom.model_name)))

#         all_loss, all_loss_train, _ = trainer.train_atom(hparams)
#         # No training loss returned?
#         self.assertEqual(0, len(all_loss))
#         self.assertEqual(0, len(all_loss_train))

#         trainer.init_flat(hparams)
#         self.assertFalse(os.path.isfile(os.path.join(hparams.out_dir, hparams.networks_dir, hparams.hparams_flat.model_name)))

#         all_loss, all_loss_train, _ = trainer.train_flat(hparams)
#         # No training loss returned?
#         self.assertEqual(0, len(all_loss))
#         self.assertEqual(0, len(all_loss_train))

#         trainer.init(hparams)
#         self.assertFalse(os.path.isfile(os.path.join(hparams.out_dir, hparams.networks_dir, hparams.model_name)))

#         _, all_loss_train, _ = trainer.train(hparams)
#         # Training loss decreases?
#         self.assertLess(all_loss_train[-1], all_loss_train[1 if hparams.start_with_test else 0],
#                         msg="Loss did not decrease over {} epochs".format(hparams.epochs))

#         shutil.rmtree(hparams.out_dir)

#     def test_train_all(self):
#         # logging.basicConfig(level=logging.INFO)

#         hparams = self._get_hparams("test_train_both")
#         hparams.hparams_flat.hparams_atom.epochs = 1
#         hparams.hparams_flat.hparams_atom.load_from_checkpoint = False
#         hparams.hparams_flat.epochs = 1
#         hparams.hparams_flat.load_from_checkpoint = False
#         hparams.epochs = 3

#         hparams.hparams_flat.atom_model_path = None
#         hparams.flat_model_path = None

#         for seed in [0]:
#             hparams.seed = seed
#             # print(seed)
#             trainer = self._get_trainer(hparams)
#             trainer.init_atom(hparams)
#             trainer.train_atom(hparams)
#             # Final atom model saved?
#             self.assertTrue(os.path.isfile(os.path.join(hparams.out_dir, hparams.networks_dir, hparams.hparams_flat.hparams_atom.model_name)))

#             trainer.init_flat(hparams)
#             trainer.train_flat(hparams)
#             # Final flat model saved?
#             self.assertTrue(os.path.isfile(os.path.join(hparams.out_dir, hparams.networks_dir, hparams.hparams_flat.model_name)))

#             trainer.init(hparams)
#             _, all_loss_train, _ = trainer.train(hparams)
#             # if all_loss_train[-1] >= all_loss_train[1 if hparams.start_with_test else 0]:
#             #     continue
#             # Training loss decreases?
#             self.assertLess(all_loss_train[-1], all_loss_train[1 if hparams.start_with_test else 0],
#                             msg="Loss did not decrease over {} epochs".format(hparams.epochs))
#             # Trained model saved.
#             self.assertTrue(os.path.isfile(os.path.join(hparams.out_dir, hparams.networks_dir, hparams.model_name)))
#             break

#         shutil.rmtree(hparams.out_dir)

#     def test_benchmark(self):
#         hparams = self._get_hparams("test_benchmark")
#         hparams.seed = 1234

#         trainer = self._get_trainer(hparams)
#         trainer.init_atom(hparams)
#         trainer.init_flat(hparams)
#         scores = trainer.flat_trainer.atom_trainer.benchmark(hparams)
#         numpy.testing.assert_almost_equal((87.312, 0.624), scores, 3)

#         scores = trainer.flat_trainer.benchmark(hparams)
#         numpy.testing.assert_almost_equal((212.879, 0.624), scores, 3)

#         hparams.load_from_checkpoint = True  # Load model.
#         trainer.init(hparams)
#         scores = trainer.benchmark(hparams)
#         numpy.testing.assert_almost_equal((1679.056, 0.604), scores, 3)

#         shutil.rmtree(hparams.out_dir)

#     def test_gen_figure(self):
#         num_test_files = 2

#         hparams = self._get_hparams("test_gen_figure")

#         trainer = self._get_trainer(hparams)
#         trainer.init(hparams)

#         with warnings.catch_warnings():
#             warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
#             trainer.gen_figure(hparams, self.id_list[:num_test_files])
#         # Check number of created files.
#         found_files = list([name for name in os.listdir(hparams.out_dir)
#                             if os.path.isfile(os.path.join(hparams.out_dir, name))
#                             and name.endswith(hparams.model_name + ".PHRASE" + hparams.gen_figure_ext)])
#         self.assertEqual(len(self.id_list[:num_test_files]), len(found_files),
#                          msg="Number of {} files in out_dir directory does not match.".format(hparams.gen_figure_ext))

#         shutil.rmtree(hparams.out_dir)

#     def test_synth_wav(self):
#         num_test_files = 2

#         hparams = self._get_hparams("test_synth_wav")
#         hparams.synth_fs = 16000
#         hparams.frame_size_ms = 5
#         hparams.synth_ext = "wav"
#         hparams.world_dir = os.path.join("integration", "fixtures", "WORLD")

#         trainer = self._get_trainer(hparams)
#         trainer.init(hparams)
#         hparams.synth_dir = hparams.out_dir
#         trainer.synth(hparams, self.id_list[:num_test_files])

#         found_files = list([name for name in os.listdir(hparams.synth_dir)
#                             if os.path.isfile(os.path.join(hparams.synth_dir, name))
#                             and name.endswith("_WORLD." + hparams.synth_ext)])
#         # Check number of created files.
#         self.assertEqual(len(self.id_list[:num_test_files]), len(found_files),
#                          msg="Number of {} files in synth_dir directory does not match.".format(hparams.synth_ext))

#         # Check readability and length of one created file.
#         raw, fs = soundfile.read(os.path.join(hparams.synth_dir, found_files[0]))
#         self.assertEqual(hparams.synth_fs, fs, msg="Desired sampling frequency of output doesn't match.")
#         labels = trainer.OutputGen[[id_name for id_name in self.id_list[:num_test_files] if id_name in found_files[0]][0]]
#         expected_length = len(raw) / hparams.synth_fs / hparams.frame_size_ms * 1000
#         self.assertTrue(abs(expected_length - len(labels)) < 10,
#                         msg="Saved raw audio file length does not roughly match length of labels.")

#         shutil.rmtree(hparams.out_dir)
