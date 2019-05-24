#!/usr/bin/python
#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#

"""
MLPG implementation based on https://github.com/MattShannon/bandmat/blob/master/example_spg.py.
Copyright 2013, 2014, 2015, 2016, 2017 Matt Shannon

Very efficient implementation of the maximum probability speech parameter
generation algorithm used in statistical speech synthesis.
(This algorithm is also sometimes known as maximum likelihood speech parameter
generation, but this is a misnomer since the quantity being maximized is not a
function of the model parameters and is therefore not a likelihood.)
Given a sequence of mean and variance parameters over time, the maximum
probability speech parameter generation algorithm computes the natural
parameters of a corresponding Gaussian distribution over trajectories, and then
computes the mean trajectory using a banded Cholesky decomposition.
The mean trajectory is also the most likely or maximum probability trajectory.
"""

import numpy as np

import bandmat as bm
import bandmat.linalg as bla


class MLPG(object):

    def build_win_mats(self, windows, frames):
        """Builds a window matrix of a given size for each window in a collection.

        `windows` specifies the collection of windows as a sequence of
        `(l, u, win_coeff)` triples, where `l` and `u` are non-negative integers
        specifying the left and right extents of the window and `win_coeff` is an
        array specifying the window coefficients.
        The returned value is a list of window matrices, one for each of the
        windows specified in `windows`.
        Each window matrix is a `frames` by `frames` Toeplitz matrix with lower
        bandwidth `l` and upper bandwidth `u`.
        The non-zero coefficients in each row of this Toeplitz matrix are given by
        `win_coeff`.
        The returned window matrices are stored as BandMats, i.e. using a banded
        representation.
        """
        win_mats = []
        for l, u, win_coeff in windows:
            assert l >= 0 and u >= 0
            assert len(win_coeff) == l + u + 1
            win_coeffs = np.tile(np.reshape(win_coeff, (l + u + 1, 1)), frames)
            win_mat = bm.band_c_bm(u, l, win_coeffs).T
            win_mats.append(win_mat)

        return win_mats

    def build_poe(self, b_frames, tau_frames, win_mats, sdw=None):
        r"""Computes natural parameters for a Gaussian product-of-experts model.

        The natural parameters (b-value vector and precision matrix) are returned.
        The returned precision matrix is stored as a BandMat.
        Mathematically the b-value vector is given as:

            b = \sum_d \transpose{W_d} \tilde{b}_d

        and the precision matrix is given as:

            P = \sum_d \transpose{W_d} \text{diag}(\tilde{tau}_d) W_d

        where $W_d$ is the window matrix for window $d$ as specified by an element
        of `win_mats`, $\tilde{b}_d$ is the sequence over time of b-value
        parameters for window $d$ as given by a column of `b_frames`, and
        $\tilde{\tau}_d$ is the sequence over time of precision parameters for
        window $d$ as given by a column of `tau_frames`.
        """
        if sdw is None:
            sdw = max([ win_mat.l + win_mat.u for win_mat in win_mats ])
        num_windows = len(win_mats)
        frames = len(b_frames)
        assert np.shape(b_frames) == (frames, num_windows)
        assert np.shape(tau_frames) == (frames, num_windows)
        assert all([ win_mat.l + win_mat.u <= sdw for win_mat in win_mats ])

        b = np.zeros((frames,))
        prec = bm.zeros(sdw, sdw, frames)

        for win_index, win_mat in enumerate(win_mats):
            bm.dot_mv_plus_equals(win_mat.T, b_frames[:, win_index], target=b)
            bm.dot_mm_plus_equals(win_mat.T, win_mat, target_bm=prec,
                                  diag=tau_frames[:, win_index])

        return b, prec

    def generation(self, features, covariance, feature_dim):
        windows = [
            (0, 0, np.array([1.0])),
            (1, 1, np.array([-0.5, 0.0, 0.5])),
            (1, 1, np.array([1.0, -2.0, 1.0])),
        ]
        num_windows = len(windows)

        frames = features.shape[0]
        smoothed_traj = np.zeros((frames, feature_dim))

        win_mats = self.build_win_mats(windows, frames)
        mean_frames = np.zeros((frames, num_windows))
        var_frames = np.zeros((frames, num_windows))

        # If feature has multiple dimension, smooth each of it.
        for d in range(feature_dim):
            var_frames[:, 0] = covariance[d, d]
            var_frames[:, 1] = covariance[feature_dim + d, feature_dim + d]
            var_frames[:, 2] = covariance[feature_dim * 2 + d, feature_dim * 2 + d]
            var_frames[0, 1] = 100000000000
            var_frames[0, 2] = 100000000000
            var_frames[-1, 1] = 100000000000
            var_frames[-1, 2] = 100000000000
            mean_frames[:, 0] = features[:, d]
            mean_frames[:, 1] = features[:, feature_dim + d]
            mean_frames[:, 2] = features[:, feature_dim * 2 + d]

            b_frames = mean_frames / var_frames
            tau_frames = 1.0 / var_frames
            b, prec = self.build_poe(b_frames, tau_frames, win_mats)
            smoothed_traj[0:frames, d] = bla.solveh(prec, b)

        return smoothed_traj
