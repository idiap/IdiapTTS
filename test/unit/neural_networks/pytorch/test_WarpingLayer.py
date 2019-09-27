#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#


import unittest

import os
import shutil
import torch
import numpy

from idiaptts.src.model_trainers.vtln.VTLNSpeakerAdaptionModelTrainer import VTLNSpeakerAdaptionModelTrainer
from idiaptts.misc.utils import makedirs_safe
from idiaptts.src.neural_networks.pytorch.models.WarpingLayer import WarpingLayer


class TestWarpingLayer(unittest.TestCase):

    out_dir = None

    @classmethod
    def setUpClass(cls):
        cls.out_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), type(cls()).__name__)
        makedirs_safe(cls.out_dir)  # Create class name directory.

    @classmethod
    def tearDownClass(cls):
        os.rmdir(cls.out_dir)  # Remove class name directory, should be empty.

    def test_compare_to_recursive_matrix(self):
        """
        Compare the element-wise computed gradient matrix with the recursively generate matrix for alphas in
        range(-alpha_range, alpha_range, precision).
        """
        precision = 0.05  # Precision used for steps in that range.
        delta = 0.05  # Allowed delta of error.

        hparams = VTLNSpeakerAdaptionModelTrainer.create_hparams()
        hparams.out_dir = os.path.join(self.out_dir, "test_compare_to_recursive_matrix")  # Add function name to path.
        wl = WarpingLayer(10, 4, hparams)
        alpha_range = wl.alpha_range  # Range of alpha to test in.
        assert (precision < 2 * alpha_range)  # Precision must fit in range.

        for alpha_value in numpy.arange(-alpha_range, alpha_range + precision, precision):
            # Alpha value which receives the final gradient.
            alpha = torch.tensor(alpha_value, requires_grad=True).type(wl.computation_dtype)
            alpha_eps = alpha.repeat([100, 1])  # Test in batch mode.

            # Compute the warp matrix for each alpha.
            warp_matrix = wl.get_warp_matrix(alpha_eps)

            # Create the reference matrix recursively for the given alpha.
            ref_matrix = wl.gen_warp_matrix_recursively(alpha)

            # Compute the error.
            numpy.testing.assert_almost_equal(warp_matrix[10].detach().numpy(), ref_matrix.detach().numpy(), 3)
            dist = (warp_matrix[10] - ref_matrix).abs()
            max_error = (dist / (ref_matrix.abs() + 1e-6)).max()
            # error = dist.sum()
            self.assertLess(max_error, delta, msg="Max error between w_matrix_3d and recursive reference is"
                                                  " {:.5f}% for alpha={:.2f}.".format(max_error * 100, alpha_value))

            # Compute the gradient ratio error.
            ref_matrix.sum().backward()
            real_grad = torch.tensor(alpha.grad)
            alpha.grad.zero_()
            warp_matrix.sum().backward()
            approx_grad = alpha.grad / len(alpha_eps)
            dist_grad = (real_grad - approx_grad).abs()
            error_ratio = (dist_grad / real_grad.abs())

            self.assertLess(error_ratio, delta, msg="Gradient error between w_matrix_3d and recursive reference is "
                                                    "{:.5f}% for alpha={:.2f}.".format(error_ratio * 100., alpha_value))

        shutil.rmtree(hparams.out_dir, ignore_errors=True)
