#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#


import unittest

import os
import shutil
import numpy
import itertools
import scipy
import soundfile
import warnings
import pyworld

from idiaptts.src.data_preparation.world.WorldFeatLabelGen import WorldFeatLabelGen
from idiaptts.misc.utils import makedirs_safe


class TestWorldFeatLabelGen(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load test data
        cls.dir_world_features = os.path.join("integration", "fixtures", "WORLD")
        cls.dir_wav = os.path.join("integration", "fixtures", "database", "wav")
        cls.id_list = cls._get_id_list()[:3]

    @staticmethod
    def _get_id_list():
        with open(os.path.join("integration", "fixtures", "database", "file_id_list.txt")) as f:
            id_list = f.readlines()
        # Trim entries in-place.
        id_list[:] = [s.strip(' \t\n\r') for s in id_list]
        return id_list

    # @classmethod
    # def tearDownClass(cls):
    #     # Remove class name directory, should be empty.
    #     os.rmdir(os.path.join(os.path.dirname(os.path.realpath(__file__)), type(cls).__name__))

    def test_load_all_with_deltas(self):

        sample = WorldFeatLabelGen.load_sample(self.id_list[0],
                                               self.dir_world_features,
                                               add_deltas=True,
                                               num_coded_sps=20)

        self.assertEqual(3 * (20 + 2) + 1, sample.shape[1], "Output shape of sample doesn't match.")

    def test_load_all(self):
        sample = WorldFeatLabelGen.load_sample(self.id_list[0],
                                               self.dir_world_features,
                                               add_deltas=False,
                                               num_coded_sps=20)

        self.assertEqual(20 + 3, sample.shape[1], "Output shape of sample doesn't match.")

    def test_load_all_but_one(self):

        sample = WorldFeatLabelGen.load_sample(self.id_list[0],
                                               self.dir_world_features,
                                               add_deltas=False,
                                               num_coded_sps=20,
                                               load_sp=False)
        self.assertEqual(3, sample.shape[1], "Output shape of sample doesn't match.")

        sample = WorldFeatLabelGen.load_sample(self.id_list[0],
                                               self.dir_world_features,
                                               add_deltas=False,
                                               num_coded_sps=20,
                                               load_lf0=False)
        self.assertEqual(20 + 2, sample.shape[1], "Output shape of sample doesn't match.")

        sample = WorldFeatLabelGen.load_sample(self.id_list[0],
                                               self.dir_world_features,
                                               add_deltas=False,
                                               num_coded_sps=20,
                                               load_vuv=False)
        self.assertEqual(20 + 2, sample.shape[1], "Output shape of sample doesn't match.")

        sample = WorldFeatLabelGen.load_sample(self.id_list[0],
                                               self.dir_world_features,
                                               add_deltas=False,
                                               num_coded_sps=20,
                                               load_bap=False)
        self.assertEqual(20 + 2, sample.shape[1], "Output shape of sample doesn't match.")

    def test_load_all_but_two(self):
        sample = WorldFeatLabelGen.load_sample(self.id_list[0],
                                               self.dir_world_features,
                                               add_deltas=False,
                                               num_coded_sps=20,
                                               load_sp=False,
                                               load_lf0=False)
        self.assertEqual(2, sample.shape[1], "Output shape of sample doesn't match.")

        sample = WorldFeatLabelGen.load_sample(self.id_list[0],
                                               self.dir_world_features,
                                               add_deltas=False,
                                               num_coded_sps=20,
                                               load_lf0=False,
                                               load_bap=False)
        self.assertEqual(20 + 1, sample.shape[1], "Output shape of sample doesn't match.")

        sample = WorldFeatLabelGen.load_sample(self.id_list[0],
                                               self.dir_world_features,
                                               add_deltas=False,
                                               num_coded_sps=20,
                                               load_vuv=False,
                                               load_bap=False)
        self.assertEqual(20 + 1, sample.shape[1], "Output shape of sample doesn't match.")

    def test_load_all_but_one_with_deltas(self):
        sample = WorldFeatLabelGen.load_sample(self.id_list[0],
                                               self.dir_world_features,
                                               add_deltas=True,
                                               num_coded_sps=20,
                                               load_sp=False)
        self.assertEqual(3 * 2 + 1, sample.shape[1], "Output shape of sample doesn't match.")

        sample = WorldFeatLabelGen.load_sample(self.id_list[0],
                                               self.dir_world_features,
                                               add_deltas=True,
                                               num_coded_sps=20,
                                               load_lf0=False)
        self.assertEqual(3 * (20 + 1) + 1, sample.shape[1], "Output shape of sample doesn't match.")

        sample = WorldFeatLabelGen.load_sample(self.id_list[0],
                                               self.dir_world_features,
                                               add_deltas=True,
                                               num_coded_sps=20,
                                               load_vuv=False)
        self.assertEqual(3 * (20 + 2), sample.shape[1], "Output shape of sample doesn't match.")

        sample = WorldFeatLabelGen.load_sample(self.id_list[0],
                                               self.dir_world_features,
                                               add_deltas=True,
                                               num_coded_sps=20,
                                               load_bap=False)
        self.assertEqual(3 * (20 + 1) + 1, sample.shape[1], "Output shape of sample doesn't match.")

    def test_load_all_but_two_with_deltas(self):
        sample = WorldFeatLabelGen.load_sample(self.id_list[0],
                                               self.dir_world_features,
                                               add_deltas=True,
                                               num_coded_sps=20,
                                               load_sp=False,
                                               load_lf0=False)
        self.assertEqual(3 * 1 + 1, sample.shape[1], "Output shape of sample doesn't match.")

        sample = WorldFeatLabelGen.load_sample(self.id_list[0],
                                               self.dir_world_features,
                                               add_deltas=True,
                                               num_coded_sps=20,
                                               load_lf0=False,
                                               load_bap=False)
        self.assertEqual(3 * 20 + 1, sample.shape[1], "Output shape of sample doesn't match.")

        sample = WorldFeatLabelGen.load_sample(self.id_list[0],
                                               self.dir_world_features,
                                               add_deltas=True,
                                               num_coded_sps=20,
                                               load_vuv=False,
                                               load_bap=False)
        self.assertEqual(3 * (20 + 1), sample.shape[1], "Output shape of sample doesn't match.")

    def test_load_norm_params_with_deltas(self):
        generator = WorldFeatLabelGen(self.dir_world_features, add_deltas=True, num_coded_sps=20)

        with self.assertRaises(ValueError, msg="Normalization parameters should be required."):
            generator[self.id_list[0]]

        generator.get_normalisation_params(self.dir_world_features)
        expected_dim = 3 * (20 + 2) + 1
        self.assertEqual(expected_dim, generator.norm_params[0].shape[0], "Loaded mean has wrong dimension.")
        self.assertEqual(expected_dim, generator.norm_params[1].shape[0], "Loaded std_dev has wrong dimension.")

        generator = WorldFeatLabelGen(self.dir_world_features, add_deltas=True, num_coded_sps=20, load_sp=False)
        generator.get_normalisation_params(self.dir_world_features)
        expected_dim = 3 * 2 + 1
        self.assertEqual(expected_dim, generator.norm_params[0].shape[0], "Loaded mean has wrong dimension.")
        self.assertEqual(expected_dim, generator.norm_params[1].shape[0], "Loaded std_dev has wrong dimension.")

        generator = WorldFeatLabelGen(self.dir_world_features, add_deltas=True, num_coded_sps=20, load_lf0=False)
        generator.get_normalisation_params(self.dir_world_features)
        expected_dim = 3 * (20 + 1) + 1
        self.assertEqual(expected_dim, generator.norm_params[0].shape[0], "Loaded mean has wrong dimension.")
        self.assertEqual(expected_dim, generator.norm_params[1].shape[0], "Loaded std_dev has wrong dimension.")

        generator = WorldFeatLabelGen(self.dir_world_features, add_deltas=True, num_coded_sps=20, load_vuv=False)
        generator.get_normalisation_params(self.dir_world_features)
        expected_dim = 3 * (20 + 2)
        self.assertEqual(expected_dim, generator.norm_params[0].shape[0], "Loaded mean has wrong dimension.")
        self.assertEqual(expected_dim, generator.norm_params[1].shape[0], "Loaded std_dev has wrong dimension.")

        generator = WorldFeatLabelGen(self.dir_world_features, add_deltas=True, num_coded_sps=20, load_bap=False)
        generator.get_normalisation_params(self.dir_world_features)
        expected_dim = 3 * (20 + 1) + 1
        self.assertEqual(expected_dim, generator.norm_params[0].shape[0], "Loaded mean has wrong dimension.")
        self.assertEqual(expected_dim, generator.norm_params[1].shape[0], "Loaded std_dev has wrong dimension.")

    def test_load_norm_params(self):
        generator = WorldFeatLabelGen(self.dir_world_features, add_deltas=False, num_coded_sps=20)

        with self.assertRaises(ValueError, msg="Normalization parameters should be required."):
            generator[self.id_list[0]]

        generator.get_normalisation_params(self.dir_world_features)
        expected_dim = 1 * (20 + 2) + 1
        self.assertEqual(expected_dim, generator.norm_params[0].shape[1], "Loaded mean has wrong dimension.")
        self.assertEqual(expected_dim, generator.norm_params[1].shape[1], "Loaded std_dev has wrong dimension.")

        generator = WorldFeatLabelGen(self.dir_world_features, add_deltas=False, num_coded_sps=20, load_sp=False)
        generator.get_normalisation_params(self.dir_world_features)
        expected_dim = 1 * 2 + 1
        self.assertEqual(expected_dim, generator.norm_params[0].shape[1], "Loaded mean has wrong dimension.")
        self.assertEqual(expected_dim, generator.norm_params[1].shape[1], "Loaded std_dev has wrong dimension.")

        generator = WorldFeatLabelGen(self.dir_world_features, add_deltas=False, num_coded_sps=20, load_lf0=False)
        generator.get_normalisation_params(self.dir_world_features)
        expected_dim = 1 * (20 + 1) + 1
        self.assertEqual(expected_dim, generator.norm_params[0].shape[1], "Loaded mean has wrong dimension.")
        self.assertEqual(expected_dim, generator.norm_params[1].shape[1], "Loaded std_dev has wrong dimension.")

        generator = WorldFeatLabelGen(self.dir_world_features, add_deltas=False, num_coded_sps=20, load_vuv=False)
        generator.get_normalisation_params(self.dir_world_features)
        expected_dim = 1 * (20 + 2)
        self.assertEqual(expected_dim, generator.norm_params[0].shape[1], "Loaded mean has wrong dimension.")
        self.assertEqual(expected_dim, generator.norm_params[1].shape[1], "Loaded std_dev has wrong dimension.")

        generator = WorldFeatLabelGen(self.dir_world_features, add_deltas=False, num_coded_sps=20, load_bap=False)
        generator.get_normalisation_params(self.dir_world_features)
        expected_dim = 1 * (20 + 1) + 1
        self.assertEqual(expected_dim, generator.norm_params[0].shape[1], "Loaded mean has wrong dimension.")
        self.assertEqual(expected_dim, generator.norm_params[1].shape[1], "Loaded std_dev has wrong dimension.")

    def test_indexing(self):
        generator = WorldFeatLabelGen(self.dir_world_features, add_deltas=True, num_coded_sps=20)
        generator.get_normalisation_params(self.dir_world_features)
        sample = generator[self.id_list[0]]
        self.assertEqual(3 * (20 + 2) + 1, sample.shape[1])

        generator = WorldFeatLabelGen(self.dir_world_features, add_deltas=False, num_coded_sps=20)
        generator.get_normalisation_params(self.dir_world_features)
        sample = generator[self.id_list[0]]
        self.assertEqual(20 + 3, sample.shape[1])

    def test_pre_and_post_processing(self):
        generator = WorldFeatLabelGen(self.dir_world_features, add_deltas=True, num_coded_sps=20)
        generator.get_normalisation_params(self.dir_world_features)

        sample = generator.load_sample(self.id_list[0],
                                       generator.dir_labels,
                                       add_deltas=generator.add_deltas,
                                       num_coded_sps=generator.num_coded_sps,
                                       load_sp=generator.load_sp,
                                       load_lf0=generator.load_lf0,
                                       load_vuv=generator.load_vuv,
                                       load_bap=generator.load_bap)
        deltas_factor = 3 if generator.add_deltas else 1
        target_sample = numpy.concatenate((sample[:, :generator.num_coded_sps],  # Frequency features.
                                           sample[:, generator.num_coded_sps * deltas_factor:generator.num_coded_sps * deltas_factor + 1],  # LF0
                                           sample[:, -deltas_factor-1:-deltas_factor],  # VUV
                                           sample[:, -deltas_factor:-deltas_factor+1]),  # BAP
                                          axis=1)

        pre_sample = generator.preprocess_sample(sample)
        post_sample = generator.postprocess_sample(pre_sample, apply_mlpg=False)

        numpy.testing.assert_almost_equal(target_sample, post_sample, decimal=5,
                                          err_msg="Original and pre- and post-processed samples are different.")

        # Repeating tests without one of the features.
        generator = WorldFeatLabelGen(self.dir_world_features, add_deltas=True, num_coded_sps=20, load_sp=False)
        generator.get_normalisation_params(self.dir_world_features)

        sample = generator.load_sample(self.id_list[0],
                                       generator.dir_labels,
                                       add_deltas=generator.add_deltas,
                                       num_coded_sps=generator.num_coded_sps,
                                       load_sp=generator.load_sp,
                                       load_lf0=generator.load_lf0,
                                       load_vuv=generator.load_vuv,
                                       load_bap=generator.load_bap)
        deltas_factor = 3 if generator.add_deltas else 1
        target_sample = numpy.concatenate((sample[:, :1],  # LF0
                                           sample[:, -deltas_factor - 1:-deltas_factor],  # VUV
                                           sample[:, -deltas_factor:-deltas_factor + 1]),  # BAP
                                          axis=1)

        pre_sample = generator.preprocess_sample(sample)
        post_sample = generator.postprocess_sample(pre_sample, apply_mlpg=False)

        numpy.testing.assert_almost_equal(target_sample, post_sample, decimal=5,
                                          err_msg="Original and pre- and post-processed samples are different.")

        generator = WorldFeatLabelGen(self.dir_world_features, add_deltas=True, num_coded_sps=20, load_lf0=False)
        generator.get_normalisation_params(self.dir_world_features)

        sample = generator.load_sample(self.id_list[0],
                                       generator.dir_labels,
                                       add_deltas=generator.add_deltas,
                                       num_coded_sps=generator.num_coded_sps,
                                       load_sp=generator.load_sp,
                                       load_lf0=generator.load_lf0,
                                       load_vuv=generator.load_vuv,
                                       load_bap=generator.load_bap)
        deltas_factor = 3 if generator.add_deltas else 1
        target_sample = numpy.concatenate((sample[:, :generator.num_coded_sps],  # Frequency features.
                                           sample[:, -deltas_factor - 1:-deltas_factor],  # VUV
                                           sample[:, -deltas_factor:-deltas_factor + 1]),  # BAP
                                          axis=1)

        pre_sample = generator.preprocess_sample(sample)
        post_sample = generator.postprocess_sample(pre_sample, apply_mlpg=False)

        numpy.testing.assert_almost_equal(target_sample, post_sample, decimal=5,
                                          err_msg="Original and pre- and post-processed samples are different.")

        generator = WorldFeatLabelGen(self.dir_world_features, add_deltas=True, num_coded_sps=20, load_vuv=False)
        generator.get_normalisation_params(self.dir_world_features)

        sample = generator.load_sample(self.id_list[0],
                                       generator.dir_labels,
                                       add_deltas=generator.add_deltas,
                                       num_coded_sps=generator.num_coded_sps,
                                       load_sp=generator.load_sp,
                                       load_lf0=generator.load_lf0,
                                       load_vuv=generator.load_vuv,
                                       load_bap=generator.load_bap)
        deltas_factor = 3 if generator.add_deltas else 1
        target_sample = numpy.concatenate((sample[:, :generator.num_coded_sps],  # Frequency features.
                                           sample[:, generator.num_coded_sps * deltas_factor:generator.num_coded_sps * deltas_factor + 1],  # LF0
                                           sample[:, -deltas_factor:-deltas_factor+1]),  # BAP
                                          axis=1)

        pre_sample = generator.preprocess_sample(sample)
        post_sample = generator.postprocess_sample(pre_sample, apply_mlpg=False)

        numpy.testing.assert_almost_equal(target_sample, post_sample, decimal=5,
                                          err_msg="Original and pre- and post-processed samples are different.")

        generator = WorldFeatLabelGen(self.dir_world_features, add_deltas=True, num_coded_sps=20, load_bap=False)
        generator.get_normalisation_params(self.dir_world_features)

        sample = generator.load_sample(self.id_list[0],
                                       generator.dir_labels,
                                       add_deltas=generator.add_deltas,
                                       num_coded_sps=generator.num_coded_sps,
                                       load_sp=generator.load_sp,
                                       load_lf0=generator.load_lf0,
                                       load_vuv=generator.load_vuv,
                                       load_bap=generator.load_bap)
        deltas_factor = 3 if generator.add_deltas else 1
        target_sample = numpy.concatenate((sample[:, :generator.num_coded_sps],  # Frequency features.
                                           sample[:, generator.num_coded_sps * deltas_factor:generator.num_coded_sps * deltas_factor + 1],  # LF0
                                           sample[:, -1:]),  # VUV
                                          axis=1)

        pre_sample = generator.preprocess_sample(sample)
        post_sample = generator.postprocess_sample(pre_sample, apply_mlpg=False)

        numpy.testing.assert_almost_equal(target_sample, post_sample, decimal=5,
                                          err_msg="Original and pre- and post-processed samples are different.")

    def test_gen_data(self):
        out_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), type(self).__name__)

        file_id_list_name = "test_id_list"
        num_coded_sps = 20
        hop_size_ms = 5

        # Generate all possible test cases.
        load_flags = ["load_sp", "load_lf0", "load_vuv", "load_bap", "add_deltas"]
        test_cases = [dict(zip(load_flags, x)) for x in itertools.product([True, False], repeat=len(load_flags))]
        # test_cases = [{"load_sp": True, "load_lf0": True, "load_vuv": True, "load_bap": True, "add_deltas": True},
        #               {"load_sp": True, "load_lf0": True, "load_vuv": True, "load_bap": True, "add_deltas": False},
        #               ...
        #              {"load_sp": False, "load_lf0": False, "load_vuv": False, "load_bap": False, "add_deltas": False}]

        for args in test_cases:
            for sp_type in ["mcep", "mfbanks"]:  # , "mgc"]:  # MGC generation takes too long to test here.
                makedirs_safe(out_dir)
                generator = WorldFeatLabelGen(out_dir,
                                              num_coded_sps=num_coded_sps,
                                              sp_type=sp_type,
                                              hop_size_ms=hop_size_ms,
                                              **args)

                return_dict, mean, std_dev = generator.gen_data(dir_in=self.dir_wav,
                                                                dir_out=out_dir,
                                                                file_id_list=file_id_list_name,
                                                                id_list=self.id_list,
                                                                return_dict=True)
                # Check length of return dict.
                self.assertEqual(len(self.id_list), len(return_dict),
                                 msg="Wrong number of entries in return_dict for {} for {}".format(sp_type, args))

                # Check dimensions of return norm params.
                if args["add_deltas"]:
                    expected_norm_entries = int(args["load_sp"]) + int(args["load_lf0"])\
                                            + int(args["load_bap"]) + int(args["load_vuv"])
                    self.assertEqual(expected_norm_entries, len(mean),
                                     msg="Wrong number of entries in means for {} for {}".format(sp_type, args))
                    self.assertEqual(expected_norm_entries, len(std_dev),
                                     msg="Wrong number of entries in std_dev for {} for {}".format(sp_type, args))
                    if args["load_sp"]:
                        self.assertEqual(3 * num_coded_sps, mean[0].shape[1],
                                         msg="Wrong number of means for {} for {}".format(sp_type, args))
                    if args["load_vuv"]:
                        vuv_index = (-1) * int(args["load_bap"]) - 1
                        self.assertEqual(0.0, mean[vuv_index][0],
                                         msg="Wrong VUV mean for {} for {}".format(sp_type, args))
                        self.assertEqual(1.0, std_dev[vuv_index][0],
                                         msg="Wrong VUV std_dev for {} for {}".format(sp_type, args))
                else:
                    expected_num_norm = (num_coded_sps * int(args["load_sp"]) + int(args["load_lf0"])
                                         + int(args["load_bap"])) * (3 if args["add_deltas"] else 1)\
                                        + int(args["load_vuv"])
                    if expected_num_norm == 0:
                        self.assertIsNone(mean)
                        self.assertIsNone(std_dev)
                    else:
                        self.assertEqual(expected_num_norm, len(mean),
                                         msg="Wrong dimension of mean for {} for {}".format(sp_type, args))
                        self.assertEqual(expected_num_norm, len(std_dev),
                                         msg="Wrong number of std_dev for {} for {}".format(sp_type, args))
                    if args["load_vuv"]:
                        vuv_index = (-1) * int(args["load_bap"]) * (3 if args["add_deltas"] else 1) - 1
                        self.assertEqual(0.0, mean[vuv_index],
                                         msg="Wrong VUV mean for {} for {}".format(sp_type, args))
                        self.assertEqual(1.0, std_dev[vuv_index],
                                         msg="Wrong VUV std_dev for {} for {}".format(sp_type, args))

                saved_as_cmp = all(value for value in args.values())  # All features present and add deltas.
                # Handle special case when everything is saved in one cmp file.
                if saved_as_cmp:
                    expected_out_dir = os.path.join(out_dir, "cmp_{}{}".format(sp_type, num_coded_sps))
                    # Count number of generated feature files.
                    found_feature_files = list([name for name in os.listdir(expected_out_dir)
                                                if os.path.isfile(os.path.join(expected_out_dir, name))
                                                and name.endswith("cmp")])
                    self.assertEqual(len(self.id_list), len(found_feature_files),
                                     msg="Wrong number of generated feature files for {} for {}".format(sp_type, args))
                    # Check if all normalisation parameter files are generated.
                    for feature, load in zip((sp_type, "lf0", "bap"),
                                             ("load_sp", "load_lf0", "load_bap")):
                        if args[load]:
                            found_norm_files = list([name for name in os.listdir(expected_out_dir)
                                                     if os.path.isfile(os.path.join(expected_out_dir, name))
                                                     and name.endswith(".bin")
                                                     and name.startswith("{}_{}".format(file_id_list_name, feature))])
                            self.assertEqual(2, len(found_norm_files),
                                             msg="Did not find two norm files for {} in {} for {}"
                                                 .format(feature, expected_out_dir, args))
                else:
                    # Handle case where no cmp file is generated but .<feature>_deltas files in each subdirectory.
                    for feature, ext, load in zip((sp_type + str(num_coded_sps), "lf0", "vuv", "bap"),
                                                  (sp_type, "lf0", "vuv", "bap"),
                                                  ("load_sp", "load_lf0", "load_vuv", "load_bap")):
                        if args[load]:
                            expected_out_dir = os.path.join(out_dir, feature)
                            # Check if feature files were generated.
                            found_feature_files = list([name for name in os.listdir(expected_out_dir)
                                                        if os.path.isfile(os.path.join(expected_out_dir, name))
                                                        and name.endswith(ext + ("_deltas" if args["add_deltas"] and feature != "vuv" else ""))])
                            self.assertEqual(len(self.id_list), len(found_feature_files),
                                             msg="Wrong number of generated feature files for {} in {} for {}"
                                             .format(feature, expected_out_dir, args))
                            # Check if normalisation parameter files were generated.
                            if feature != "vuv":
                                found_norm_files = list([name for name in os.listdir(expected_out_dir)
                                                         if os.path.isfile(os.path.join(expected_out_dir, name))
                                                         and name.endswith(".bin")
                                                         and name.startswith(file_id_list_name)])
                                self.assertEqual(2, len(found_norm_files),
                                                 msg="Did not find two norm files for {} in {} for {}"
                                                 .format(feature, expected_out_dir, args))

                shutil.rmtree(out_dir)

    @staticmethod
    def _plot_power_sp(ground_truth, reconstruction, sp_type=None, num_coded_sps=None, label_ground_truth=None, label_reconstruction=None):

        if not label_ground_truth:
            label_ground_truth = "WORLD amplitude spectrum in dB"
        if not label_reconstruction:
            label_reconstruction = "Reconstruction from {} {} bins in dB".format(num_coded_sps, sp_type)
        # DEBUG
        import librosa
        import matplotlib.pyplot as plt
        frame_idx = 200
        plt.plot(librosa.power_to_db(ground_truth.T, top_db=None)[:, frame_idx], "b-", linewidth=2.0, label=label_ground_truth)
        plt.plot(librosa.power_to_db(reconstruction.T, top_db=None)[:, frame_idx], "r-", linewidth=2.0, label=label_reconstruction)
        # plt.plot(20/np.log(20)*envelope, "r-", lindewidth=3.0, label="Reconstruction")
        plt.xlabel("frequency bin")
        plt.ylabel("log amplitude")
        plt.legend()
        plt.show()

    @staticmethod
    def _plot_waveform(waveforms, normalise=True):
        if waveforms is tuple:  # Only one waveform to plot
            waveforms = [waveforms]

        assert len(waveforms[0]) == 3, "Each entry in input should be (waveform, fs, label)."

        import librosa.display
        import matplotlib.pyplot as plt
        plt.figure()

        plt_idx = 1
        ax = None
        for y, fs, label in waveforms:
            if ax is None:
                ax = plt.subplot(len(waveforms), 1, plt_idx)
            else:
                plt.subplot(len(waveforms), 1, plt_idx, sharex=ax, sharey=ax)
            if normalise:
                if y.min() > 0:  # Signal is only positive.
                    y /= y.max()
                    y -= 0.5
                else:
                    y /= max(abs(y.min()), y.max())

            librosa.display.waveplot(y, sr=fs)
            plt.title(label)
            if plt_idx < len(waveforms):
                plt.xlabel('')
            plt_idx += 1

        plt.tight_layout()
        plt.show()

    def test_sp_reconstruction(self):

        # out_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), type(self).__name__)
        # if not os.path.isdir(out_dir):
        #     os.mkdir(out_dir)
        hop_size_ms = 5
        preemphasis = 0.97

        # Load raw audio data.
        file_name = self.id_list[0]
        audio_name = os.path.join(self.dir_wav, file_name + ".wav")
        raw, fs = WorldFeatLabelGen.get_raw(audio_name, preemphasis)
        raw_no_preemphasis, fs = WorldFeatLabelGen.get_raw(audio_name, preemphasis=0.0)

        # Compare extracted amplitude spectrum from librosa and World.
        librosa_amp_sp = WorldFeatLabelGen.librosa_extract_amp_sp(raw, fs, hop_size_ms=5)
        world_amp_sp, lf0, vuv, bap = WorldFeatLabelGen.world_extract_features(raw, fs, hop_size_ms=5)
        # TestWorldFeatLabelGen._plot_power_sp(world_amp_sp**2, librosa_amp_sp**2,
        #                                      label_ground_truth="WORLD amp sp in dB",
        #                                      label_reconstruction="Librosa amp sp in dB")
        self.assertGreater(5000, ((librosa_amp_sp - world_amp_sp) ** 2).sum(),
                           msg="Librosa and WORLD amplitude spectrum don't seem to be in the same domain.")

        # Compare respective reconstruction.
        librosa_raw_gl = WorldFeatLabelGen.amp_sp_to_raw(librosa_amp_sp, fs, hop_size_ms, preemphasis)
        librosa_raw_gl /= max(librosa_raw_gl.max(), abs(librosa_raw_gl.min()))  # Scale to [-1, 1]

        world_raw_world_vocoder = WorldFeatLabelGen.world_features_to_raw(world_amp_sp, lf0, vuv, bap, fs)
        world_raw_world_vocoder /= max(world_raw_world_vocoder.max(), abs(world_raw_world_vocoder.min()))  # To [-1, 1]

        # # DEBUG: Reconstructing waveform from librosa amp sp with World vocder and
        # #        reconstructing wavefrom from world amp sp with GL. Both doesn't work.
        # librosa_raw_world_vocoder = WorldFeatLabelGen.world_features_to_raw(librosa_amp_sp, lf0, vuv, bap, fs)
        # librosa_raw_world_vocoder /= max(librosa_raw_world_vocoder.max(), abs(librosa_raw_world_vocoder.min()))
        # world_raw_gl = WorldFeatLabelGen.amp_sp_to_raw(world_amp_sp, fs, hop_size_ms, preemphasis)
        # world_raw_gl /= max(world_raw_gl.max(), abs(world_raw_gl.min()))  # Scale to [-1, 1]
        # TestWorldFeatLabelGen._plot_waveform([(raw_no_preemphasis, fs, "Original"),
        #                                       (librosa_raw_gl, fs, "Reconstruction from librosa amp sp with GL"),
        #                                       (librosa_raw_world_vocoder, fs, "Reconstruction from librosa amp sp with WORLD vocoder."),
        #                                       (world_raw_gl, fs, "Reconstruction from WORLD amp sp with GL"),
        #                                       (world_raw_world_vocoder, fs, "Reconstruction from WORLD amp sp with WORLD vocoder.")],
        #                                      normalise=True)
        # soundfile.write(os.path.join(out_dir, "{}_{}_GL.wav".format(file_name, "librosa_raw")), librosa_raw_gl, fs)
        # soundfile.write(os.path.join(out_dir, "{}_{}_World_vocoder.wav".format(file_name, "librosa_raw")), librosa_raw_world_vocoder, fs)
        # soundfile.write(os.path.join(out_dir, "{}_{}_GL.wav".format(file_name, "world_raw")), world_raw_gl, fs)
        # soundfile.write(os.path.join(out_dir, "{}_{}_World_vocoder.wav".format(file_name, "world_raw")), world_raw_world_vocoder, fs)

        # # DEBUG: Extraction and reconstruction with world has major artifacts.
        # direct_world_raw = scipy.signal.lfilter([1], [1, -preemphasis],
        #                                         pyworld.synthesize(*pyworld.wav2world(raw, fs), fs)
        #                                         .astype(numpy.float32))
        # soundfile.write(os.path.join(out_dir, "{}_{}.wav".format(file_name, "direct_world_vocoder")), direct_world_raw, fs)

        self.assertGreater(10000, ((raw_no_preemphasis[:len(librosa_raw_gl)] - librosa_raw_gl) ** 2).sum(),
                           msg="Librosa amp spectrum GL reconstruction significantly different from original.")
        self.assertGreater(10000, ((raw_no_preemphasis[:len(world_raw_world_vocoder)]
                                    - world_raw_world_vocoder[:len(raw_no_preemphasis)]) ** 2).sum(),
                           msg="World amp spectrum World vocoder reconstruction significantly different from original.")

    def test_coded_sp_reconstruction(self):
        # import librosa

        out_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), type(self).__name__)
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)
        num_coded_sps = 80
        hop_size_ms = 5
        preemphasis = 0.97

        # Load raw audio data.
        file_name = self.id_list[0]
        audio_name = os.path.join(self.dir_wav, file_name + ".wav")
        raw, fs = WorldFeatLabelGen.get_raw(audio_name, preemphasis)
        raw_no_preemphasis, fs = WorldFeatLabelGen.get_raw(audio_name, preemphasis=0.0)

        # Compare extracted amplitude spectrum from librosa and World.
        librosa_amp_sp = WorldFeatLabelGen.librosa_extract_amp_sp(raw, fs, hop_size_ms=5)
        world_amp_sp, lf0, vuv, bap = WorldFeatLabelGen.world_extract_features(raw, fs, hop_size_ms=5)

        # # DEBUG: Reconstruction from mfbanks impossible.
        # y, fs = librosa.load(audio_name, sr=16000)
        # y = numpy.append(y[0], y[1:] - preemphasis * y[:-1])  # Preemphasis
        # S = numpy.abs(librosa.stft(raw, n_fft=1024, hop_length=int(fs * hop_size_ms / 1000.)))
        # mel_spec = librosa.feature.melspectrogram(S=S, sr=fs, n_mels=num_coded_sps)
        # S_inv = librosa.feature.inverse.mel_to_stft(mel_spec, sr=fs, n_fft=1024)
        # TestWorldFeatLabelGen._plot_power_sp(S.T**2, S_inv.T**2, sp_type="mfbanks", num_coded_sps=num_coded_sps,
        #                                      label_ground_truth="Librosa amplitude spectrum in dB")
        # y_inv_S = librosa.griffinlim(S, hop_length=int(fs * hop_size_ms / 1000.))
        # y_inv = librosa.griffinlim(S_inv, hop_length=int(fs * hop_size_ms / 1000.))
        # TestWorldFeatLabelGen._plot_waveform([(y, fs, "Original"), (y_inv_S, fs, "GL"), (y_inv, fs, "mfbanks GL")])
        # y_inv = scipy.signal.lfilter([1], [1, -preemphasis], y_inv)  # de-preemphasis
        # librosa.output.write_wav(os.path.join(out_dir, "{}_{}_griffinlim.wav".format(file_name, "test")),
        #                          y_inv, fs, norm=True)
        # # soundfile.write(os.path.join(out_dir, "{}_{}_griffinlim.wav".format(file_name, "test")), y_inv, fs)

        # for sp_type in ["mfbanks", "mcep"]:  #, "mgc"]:

        # # Check mfbanks reconstruction.
        # coded_sp = WorldFeatLabelGen.extract_mfbanks(fs=fs, amp_sp=librosa_amp_sp, hop_size_ms=hop_size_ms,
        #                                              num_coded_sps=num_coded_sps)
        # reconstruction = WorldFeatLabelGen.mfbanks_to_amp_sp(coded_sp, fs)
        # # TestWorldFeatLabelGen._plot_power_sp(librosa_amp_sp**2, reconstruction**2, sp_type="mfbanks",
        # #                                      num_coded_sps=num_coded_sps,
        # #                                      label_ground_truth="Librosa amplitude spectrum in dB")
        # self.assertGreater(500, ((librosa_amp_sp - reconstruction)**2).sum(),
        #                    msg="Mfbanks reconstruction doesn't seem to be in the same domain.")

        # Check mcep reconstruction.
        mcep = WorldFeatLabelGen.extract_mcep(amp_sp=world_amp_sp,
                                              num_coded_sps=num_coded_sps,
                                              mgc_alpha=WorldFeatLabelGen.fs_to_mgc_alpha(fs))
        reconstruction = WorldFeatLabelGen.mcep_to_amp_sp(mcep, fs)
        # TestWorldFeatLabelGen._plot_power_sp(world_amp_sp**2, reconstruction**2, sp_type="mcep",
        #                                      num_coded_sps=num_coded_sps,
        #                                      label_ground_truth="World amplitude spectrum in dB")
        self.assertGreater(50, ((world_amp_sp - reconstruction) ** 2).sum(),
                           msg="Mfbanks reconstruction doesn't seem to be in the same domain.")

        # Check mcep reconstruction.
        mgc = WorldFeatLabelGen.extract_mgc(raw, fs=fs, num_coded_sps=num_coded_sps,
                                            frame_hop_ms=hop_size_ms,
                                            mgc_alpha=WorldFeatLabelGen.fs_to_mgc_alpha(fs))
        reconstruction = WorldFeatLabelGen.mgc_to_amp_sp(mgc, fs)
        # TestWorldFeatLabelGen._plot_power_sp(librosa_amp_sp ** 2, reconstruction ** 2, sp_type="mgc",
        #                                      num_coded_sps=num_coded_sps,
        #                                      label_ground_truth="Librosa amplitude spectrum in dB")
        self.assertGreater(1000, ((librosa_amp_sp - reconstruction) ** 2).sum(),
                           msg="MGC reconstruction doesn't seem to be in the same domain.")

        # # Compare MGC and MCep which should be in the same domain.
        # librosa_mcep = WorldFeatLabelGen.extract_mcep(librosa_amp_sp, num_coded_sps,
        #                                               WorldFeatLabelGen.fs_to_mgc_alpha(fs))
        # TestWorldFeatLabelGen._plot_power_sp(numpy.exp(librosa_mcep), numpy.exp(mgc),
        #                                      label_ground_truth="MCep from librosa amp sp",
        #                                      label_reconstruction="MGC from SPTK amp sp")
        # # self.assertGreater(100000, ((librosa_mcep - mgc) ** 2).sum(),
        # #                    msg="Extracted MGC and MCep do not seem to be in the same domain.")
        # librosa_mcep_reconstruction = WorldFeatLabelGen.mcep_to_amp_sp(librosa_mcep, fs)
        # TestWorldFeatLabelGen._plot_power_sp(librosa_mcep_reconstruction, reconstruction,
        #                                      label_ground_truth="Amp sp from MCep from librosa amp sp",
        #                                      label_reconstruction="Amp sp from MGC from SPTK amp sp")
        # TestWorldFeatLabelGen._plot_power_sp(librosa_amp_sp, reconstruction, sp_type="mgc",
        #                                      label_ground_truth="Librosa amp sp",
        #                                      label_reconstruction="Amp sp from MGC from SPTK amp sp")
        # TestWorldFeatLabelGen._plot_power_sp(librosa_amp_sp, librosa_mcep_reconstruction, sp_type="mcep",
        #                                      label_ground_truth="Librosa amp sp",
        #                                      label_reconstruction="Amp sp from MCep from librosa amp sp")
