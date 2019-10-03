#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#


import unittest

import os
import shutil
import numpy
import soundfile
import warnings

from idiaptts.src.data_preparation.world.WorldFeatLabelGen import WorldFeatLabelGen


class TestWorldFeatLabelGen(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load test data
        cls.dir_world_features = os.path.join("integration", "fixtures", "WORLD")
        cls.id_list = cls._get_id_list()

    @staticmethod
    def _get_id_list():
        with open(os.path.join("integration", "fixtures", "database", "file_id_list.txt")) as f:
            id_list = f.readlines()
        # Trim entries in-place.
        id_list[:] = [s.strip(' \t\n\r') for s in id_list]
        return id_list

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
