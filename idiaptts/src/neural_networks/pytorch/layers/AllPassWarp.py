#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#


import logging
import torch
import math
import numpy as np
from functools import reduce

from torch import nn

from idiaptts.misc.utils import ncr


class AllPassWarp(nn.Module):

    computation_dtype = 'torch.FloatTensor'  # torch.float32 cannot be pickled.

    def __init__(self, warp_matrix_size):
        super().__init__()

        self.warp_matrix_size = warp_matrix_size
        self.max_polynomial = 2 * self.warp_matrix_size

        self.register_buffer("w_matrix_3d", self.gen_w_matrix_3d(), persistent=False)
        # self.w_matrix_3d[1:, 0] = 0.0  # Don't change c_0.
        # self.all_pass_warp = AllPassWarpMatrixGenerator()

        # self.compare_with_recursive(0.2)

    def init_hidden(self, batch_size=1):
        return None

    def gen_w_matrix_3d(self):
        """
        Computes the entries with the formula for m-th row and k-th column:
        A(m, k) = 1/(k-1)! * sum_{n=max(0, k-m}}^k (k choose n) * (m+n-1)! / (m+n-k)! * (-1)^{n+k+m} alpha^{2n+m-k}
        The entries are stored as a vector corresponding to the polynomials of alpha (1, alpha, alpha^2,..., alpha^{M-1}).

        :return:     Values in warping matrix.
        """
        grad_matrix = np.zeros((self.warp_matrix_size, self.warp_matrix_size, self.max_polynomial),
                               dtype=np.float64)  # (np.float64 if self.n >= 32 else np.float32))  # 32! = 2.6E35

        grad_matrix[0, 0, 0] = 1.0
        max_degree = 0
        for m in range(0, self.warp_matrix_size):
            for k in range(1, self.warp_matrix_size):
                k_fac = 1 / math.factorial(k - 1) if k > 0 else 1
                for n in range(max(0, k - m), k + 1):
                    w = ncr(k, n) * math.pow(-1, n + m + k)  # * (2 * n + m - k)
                    w_fac = math.factorial(m + n - 1) if m + n - 1 > 0 else 1
                    w_fac /= math.factorial(m + n - k) if m + n - k > 0 else 1
                    w *= w_fac * k_fac
                    # if w != 0.0:
                    degree = 2 * n + m - k  # - 1
                    if degree < self.max_polynomial:
                        grad_matrix[m, k, degree] = w
                        # if degree > max_degree:
                        #     max_degree = degree
                        #     max_w = w
                        #     max_m = m
                        #     max_k = k

        # Deal with hugh factorials.
        # w_matrix_3d[w_matrix_3d == np.inf] = np.finfo('f').max
        # w_matrix_3d[w_matrix_3d == -np.inf] = -np.finfo('f').max

        grad_matrix = torch.from_numpy(grad_matrix).type(self.computation_dtype)
        grad_matrix.requires_grad = False
        grad_matrix = torch.transpose(grad_matrix, 0, 1).contiguous()

        return grad_matrix

    def gen_warp_matrix_recursively(self, alpha, requires_recursive_grad=True):
        m = [[torch.empty((1,), dtype=alpha.dtype, requires_grad=requires_recursive_grad)
              for x in range(self.warp_matrix_size)] for y in range(self.warp_matrix_size)]
        n = self.warp_matrix_size
        m[0][0] = torch.ones((1,), dtype=alpha.dtype, requires_grad=requires_recursive_grad)
        for r in range(1, n):
            m[r][0] = m[r - 1][0] * alpha
        for c in range(1, n):
            m[0][c] = torch.zeros((1,), dtype=alpha.dtype, requires_grad=requires_recursive_grad)  # Fix for transpose.
            for r in range(1, n):
                m[r][c] = m[r - 1][c - 1] + alpha * (m[r - 1][c] - m[r][c - 1])

        grad_matrix = torch.cat([torch.cat(x) for x in m]).view(self.warp_matrix_size, self.warp_matrix_size)
        return grad_matrix

    def compare_with_recursive(self, alpha_range, precision=0.05, delta=0.001):
        """
        Compare the element-wise computed gradient matrix with the recursively generate matrix for alphas in
        range(-alpha_range, alpha_range, precision).

        :param alpha_range:           Range of alpha to test in.
        :param precision:             Precision used for steps in that range.
        :param delta:                 Allowed delta of error.

        :return:
        """
        assert(precision < 2 * alpha_range)  # Precision must fit in range.

        for alpha_value in np.arange(-alpha_range, alpha_range + precision, precision):
            # Alpha value which receives the final gradient.
            alpha = torch.tensor(alpha_value, dtype=self.w_matrix_3d.dtype, requires_grad=True)
            alpha_eps = alpha
            alpha_eps = alpha_eps.repeat([1000, 1])

            # Compute the warp matrix for each alpha.
            warp_matrix = self.get_warp_matrix(alpha_eps)

            # Create the reference matrix recursively for the given alpha.
            ref_matrix = self.gen_warp_matrix_recursively(alpha)

            # Compute the error.
            dist = (warp_matrix[10] - ref_matrix).abs()
            max_error = (dist / (ref_matrix.abs() + 1e-6)).max()
            error = dist.sum()

            err_msg = "Max error between w_matrix_3d and recursive reference is (cum: {:.5f}, max: {:.5f}) {:.5f}% for alpha={:.2f}."\
                .format(error, dist.max(), max_error * 100, alpha_value)
            logging.error(err_msg)
            # if max_error > delta:
            #     raise ValueError(err_msg)

            # Compute the gradient ratio error.
            ref_matrix.sum().backward()
            real_grad = torch.tensor(alpha.grad)
            alpha.grad.zero_()
            warp_matrix.sum().backward()
            approx_grad = alpha.grad / len(alpha_eps)
            dist_grad = (real_grad - approx_grad).abs()
            error_ratio = (dist_grad / real_grad.abs())

            err_msg = "Gradient error between w_matrix_3d and recursive reference is ({:.5f})  {:.5f}% for alpha={:.2f}."\
                .format(dist_grad, error_ratio * 100., alpha_value)
            logging.error(err_msg)
            # if error_ratio > delta:
            #     raise ValueError(err_msg)

        return True

    def forward(self, in_tensor, alphas, out_tensor=None):

        if out_tensor is None:
            out_tensor = torch.empty(in_tensor.shape,
                                     dtype=in_tensor.dtype,
                                     device=in_tensor.device,
                                     requires_grad=False)
        time_batch_dim = in_tensor.shape[:2]

        combined_alphas = AllPassWarp.combine_warping_parameters(alphas)
        warp_matrix = self.get_warp_matrix(combined_alphas.contiguous().view(-1, 1))

        in_tensor = in_tensor.view(warp_matrix.shape[0], 1, *in_tensor.shape[2:]) # TODO: .clone()  # Merge time and batch axis.

        in_tensor[:, :, 0:3 * self.warp_matrix_size:self.warp_matrix_size] /= 2.  # Adaptation for single-sided spectrogram.
        for start_index in range(0, in_tensor.shape[2] // self.warp_matrix_size):
            feature = in_tensor[:, :, start_index * self.warp_matrix_size:(start_index + 1) * self.warp_matrix_size]

            # (TB x 1 x N) * (TB x N x N) -> (TB x 1 x N) -> (T x B x N)
            feature_warped = torch.bmm(feature, warp_matrix).view(*time_batch_dim, self.warp_matrix_size)  # Split into time and batch axis again.

            out_tensor[:, :, start_index * self.warp_matrix_size:(start_index + 1) * self.warp_matrix_size] = feature_warped

        out_tensor[:, :, 0:3 * self.warp_matrix_size:self.warp_matrix_size] *= 2.  # Adaptation for single-sided spectrogram.

        return out_tensor, combined_alphas

    @staticmethod
    def combine_warping_parameters(alphas):
        if type(alphas) in [list, tuple]:
            return reduce(AllPassWarp._add_warping_parameters, alphas)
        else:
            return alphas

    @staticmethod
    def _add_warping_parameters(alpha_1, alpha_2):
        return (alpha_1 + alpha_2) / (1 + alpha_1 * alpha_2)

    def get_warp_matrix(self, alphas):
        """
        Compute warping matrix for vector of alphas.

        :param alphas:       Vector of alphas with time and batch dimension merged (TB x 1).
        :return:             Warping matrix for each alpha value with merged time and batch dimension (TB x n x n).
        """

        alpha_vec = AllPassWarp._get_polynomial_map(alphas, self.max_polynomial - 1)

        # Do a batched matrix multiplication to get the elements of the warp matrix.
        # alpha_vec = alpha_vec.unsqueeze(-1)
        # grad_matrix_flat = self.w_matrix_3d.view(self.warp_matrix_size * self.warp_matrix_size, self.w_matrix_3d.shape[-1])  # n x n x 2n -> n^2 x 2n
        # grad_matrix_ext_flat = grad_matrix_flat.expand(alpha_vec.shape[0], *grad_matrix_flat.shape[0:])  # TB x n^2 x 2n
        # warp_matrix = torch.matmul(grad_matrix_ext_flat, alpha_vec).view(-1, self.warp_matrix_size, self.warp_matrix_size)  # TB x n x n

        # print("w_matrix_3d device: " + str(self.w_matrix_3d.device))
        warp_matrix = torch.einsum("ijk,lk->lij", [self.w_matrix_3d, alpha_vec])

        return warp_matrix

    @staticmethod
    def _get_polynomial_map(alphas, max_polynomial):
        # alpha_list = [torch.ones(alphas.shape, dtype=alphas.dtype, device=alphas.device, requires_grad=True)]
        # for i in range(1, max_polynomial + 1):
        #     alpha_list.append(alpha_list[i - 1] * alphas)
        # alpha_vec = torch.cat(alpha_list, dim=1)  # T x 2n x 1

        # return alpha_vec

        # print("alphas device: " + str(alphas.device))
        a = alphas.expand(alphas.shape[0], max_polynomial).cumprod(dim=-1)
        return torch.cat([
            torch.ones(a.shape[0], 1, device=alphas.device, dtype=alphas.dtype),
            a
        ], dim=-1)
