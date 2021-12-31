# #
# # Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# # Written by Bastian Schnell <bastian.schnell@idiap.ch>
# #


# import unittest

# import os
# import shutil
# import numpy
# import soundfile
# import warnings
# from functools import partial

# from idiaptts.src.model_trainers.WaveNetVocoderTrainer import WaveNetVocoderTrainer
# from idiaptts.src.data_preparation.audio.RawWaveformLabelGen import RawWaveformLabelGen


# class TestWaveNetVocoderTrainer(unittest.TestCase):

#     max_auto_regressive_frames = 50

#     @classmethod
#     def setUpClass(cls):
#         # Load test data
#         cls.dir_world_features = os.path.join("integration", "fixtures", "WORLD")
#         cls.id_list = cls._get_id_list()

#     @classmethod
#     def tearDownClass(cls):
#         hparams = cls._get_hparams(cls())
#         os.rmdir(hparams.out_dir)  # Remove class name directory, should be empty.

#     def _get_hparams(self):
#         hparams = WaveNetVocoderTrainer.create_hparams()
#         # General parameters
#         hparams.voice = "full"
#         hparams.data_dir = os.path.realpath(os.path.join("integration", "fixtures", "database", "wav"))
#         hparams.out_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), type(self).__name__)

#         hparams.frame_size_ms = 5
#         hparams.num_coded_sps = 20
#         hparams.seed = 1

#         # Training parameters.
#         hparams.epochs = 3
#         hparams.use_gpu = False
#         hparams.model_type = "r9y9WaveNet"
#         hparams.batch_size_train = 2
#         hparams.batch_size_val = 2
#         hparams.batch_size_synth = 1
#         hparams.use_saved_learning_rate = True
#         hparams.optimiser_args["lr"] = 0.001
#         hparams.model_name = "test_model.nn"
#         hparams.epochs_per_checkpoint = 2

#         # Create a very small network.
#         hparams.input_type = "mulaw-quantize"
#         hparams.quantize_channels = 128
#         hparams.mu = 127
#         hparams.out_channels = hparams.quantize_channels
#         hparams.cin_channels = hparams.num_coded_sps + 3
#         hparams.upsample_conditional_features = True
#         hparams.upsample_scales = [1]
#         hparams.len_in_out_multiplier = 1  # Has to match the upsampling.
#         hparams.layers = 4
#         hparams.stacks = 2
#         hparams.residual_channels = 2
#         hparams.gate_channels = 2
#         hparams.skip_out_channels = 2
#         hparams.kernel_size = 2
#         hparams.add_hparam("max_input_train_sec", 0.2)
#         hparams.add_hparam("max_input_test_sec", 0.1)

#         return hparams

#     @staticmethod
#     def _get_id_list():
#         with open(os.path.join("integration", "fixtures", "database", "file_id_list.txt")) as f:
#             id_list = f.readlines()
#         # Trim entries in-place.
#         id_list[:] = [s.strip(' \t\n\r') for s in id_list]

#         # The WaveNet implementation requires the full path.
#         for index, id_name in enumerate(id_list):
#             id_list[index] = os.path.join(os.path.realpath(os.path.join("integration", "fixtures", "database", "wav")),
#                                           id_name + ".wav")

#         return id_list

#     def test_init(self):
#         hparams = self._get_hparams()
#         hparams.out_dir = os.path.join(hparams.out_dir, "test_init")  # Add function name to path.

#         trainer = WaveNetVocoderTrainer(self.dir_world_features, self.id_list, hparams)
#         trainer.init(hparams)

#         shutil.rmtree(hparams.out_dir)

#     def test_train(self):
#         hparams = self._get_hparams()
#         hparams.out_dir = os.path.join(hparams.out_dir, "test_train")  # Add function name to path.
#         hparams.seed = 1
#         hparams.use_best_as_final_model = False

#         trainer = WaveNetVocoderTrainer(self.dir_world_features, self.id_list, hparams)
#         trainer.init(hparams)
#         _, all_loss_train, _ = trainer.train(hparams)

#         # Training loss decreases?
#         self.assertLess(all_loss_train[-1], all_loss_train[1 if hparams.start_with_test else 0],
#                         msg="Loss did not decrease over {} epochs".format(hparams.epochs))

#         shutil.rmtree(hparams.out_dir)

#     @staticmethod
#     def trimming_batch_collate_fn(batch, common_divisor=1, batch_first=False, use_cond=True, one_hot_target=True):
#         """A function that trims the inputs so that auto-regressive synthesis is fast for the test case."""
#         inputs, targets, seq_length_input, *misc = WaveNetVocoderTrainer.prepare_batch(batch,
#                                                                                        common_divisor,
#                                                                                        batch_first,
#                                                                                        use_cond,
#                                                                                        one_hot_target)
#         if inputs is not None:
#             inputs = inputs[..., :TestWaveNetVocoderTrainer.max_auto_regressive_frames]
#             seq_length_input[seq_length_input > TestWaveNetVocoderTrainer.max_auto_regressive_frames] = \
#                 TestWaveNetVocoderTrainer.max_auto_regressive_frames
#         return (inputs, targets, seq_length_input, *misc)

#     def test_gen_figure(self):
#         num_test_files = 2

#         hparams = self._get_hparams()
#         hparams.out_dir = os.path.join(hparams.out_dir, "test_gen_figure")  # Add function name to path
#         hparams.model_name = "r9y9_wavenet_in23_out128.nn"
#         hparams.model_path = os.path.join("integration", "fixtures", hparams.model_name)
#         hparams.batch_size_gen_figure = 1

#         trainer = WaveNetVocoderTrainer(self.dir_world_features, self.id_list, hparams)
#         trainer.batch_collate_fn = partial(self.trimming_batch_collate_fn,
#                                            use_cond=hparams.use_cond,
#                                            one_hot_target=True)

#         trainer.init(hparams)

#         with warnings.catch_warnings():
#             warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
#             trainer.gen_figure(hparams, self.id_list[:num_test_files])
#         # Check number of created files.
#         found_files = list([name for name in os.listdir(hparams.out_dir)
#                             if os.path.isfile(os.path.join(hparams.out_dir, name))
#                             and name.endswith(hparams.model_name + ".Raw" + hparams.gen_figure_ext)])
#         self.assertEqual(len(self.id_list[:num_test_files]), len(found_files),
#                          msg="Number of {} files in out_dir directory does not match.".format(hparams.gen_figure_ext))

#         shutil.rmtree(hparams.out_dir)

#     def test_synth_wav(self):
#         num_test_files = 2

#         hparams = self._get_hparams()
#         hparams.out_dir = os.path.join(hparams.out_dir, "test_synth_wav")  # Add function name to path
#         hparams.model_name = "r9y9_wavenet_in23_out128.nn"
#         hparams.model_path = os.path.join("integration", "fixtures", hparams.model_name)
#         hparams.synth_ext = "wav"

#         trainer = WaveNetVocoderTrainer(self.dir_world_features, self.id_list, hparams)
#         trainer.batch_collate_fn = partial(self.trimming_batch_collate_fn,
#                                            use_cond=hparams.use_cond,
#                                            one_hot_target=True)

#         trainer.init(hparams)
#         hparams.synth_dir = hparams.out_dir
#         trainer.synth(hparams, self.id_list[:num_test_files])

#         found_files = list([name for name in os.listdir(hparams.synth_dir)
#                             if os.path.isfile(os.path.join(hparams.synth_dir, name))
#                             and name.endswith(hparams.model_name + "." + hparams.synth_ext)])
#         # Check number of created files.
#         self.assertEqual(len(self.id_list[:num_test_files]), len(found_files),
#                          msg="Number of {} files in synth_dir directory does not match.".format(hparams.synth_ext))

#         # Check readability and length of one created file.
#         raw, fs = soundfile.read(os.path.join(hparams.synth_dir, found_files[0]))
#         self.assertEqual(hparams.synth_fs, fs, msg="Desired sampling frequency of output doesn't match.")
#         # files = [id_name for id_name in self.id_list[:num_test_files]
#         # if "{}_{}.{}".format(os.path.splitext(os.path.basename(id_name))[0],
#         #                      hparams.model_name, hparams.synth_ext) in found_files]
#         # labels = trainer.OutputGen[files[0]]

#         self.assertEqual(len(raw), TestWaveNetVocoderTrainer.max_auto_regressive_frames,
#                          msg="Saved raw audio file length does not match num_synth_frames.")

#         shutil.rmtree(hparams.out_dir)

#     def test_copy_synth(self):
#         num_test_files = 2

#         hparams = self._get_hparams()
#         hparams.out_dir = os.path.join(hparams.out_dir, "test_copy_synth")  # Add function name to path
#         hparams.synth_ext = "wav"

#         trainer = WaveNetVocoderTrainer(self.dir_world_features, self.id_list, hparams)
#         trainer.init(hparams)
#         hparams.synth_dir = hparams.out_dir
#         trainer.copy_synth(hparams, self.id_list[:num_test_files])

#         found_files = list([name for name in os.listdir(hparams.synth_dir)
#                             if os.path.isfile(os.path.join(hparams.synth_dir, name))
#                             and name.endswith("_ref." + hparams.synth_ext)])
#         # Check number of created files.
#         self.assertEqual(len(self.id_list[:num_test_files]), len(found_files),
#                          msg="Number of {} files in synth_dir directory does not match.".format(hparams.synth_ext))

#         # Check readability and length of one created file.
#         raw, fs = soundfile.read(os.path.join(hparams.synth_dir, found_files[0]))
#         self.assertEqual(hparams.synth_fs, fs, msg="Desired sampling frequency of output doesn't match.")

#         self.assertTrue((raw == RawWaveformLabelGen.load_sample(
#                                   os.path.join(os.path.dirname(self.id_list[0]), found_files[0].split('_')[0] + ".wav"),
#                                   hparams.frame_rate_output_Hz)).all(),
#                         msg="Saved raw audio file does not match reference.")

#         shutil.rmtree(hparams.out_dir)
