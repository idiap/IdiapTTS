#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#

"""Module description.

"""

# System imports.
import os
import logging
import math
import copy
import numpy as np
from timeit import default_timer as timer
from datetime import timedelta

# Third-party imports.
import torch
import torch.nn as nn

# Local source tree imports.
from idiaptts.misc.utils import ncr, makedirs_safe
from idiaptts.src.Synthesiser import Synthesiser


class WarpingLayer(nn.Module):
    IDENTIFIER = "VTLN"
    logger = logging.getLogger(__name__)

    def __init__(self, dim_in, dim_out, hparams):
        super().__init__()

        # Store parameters.
        self.use_gpu = hparams.use_gpu
        self.dim_in = dim_in
        self.dim_out = dim_out
        norm_params_dim = hparams.num_coded_sps * (3 if hparams.add_deltas else 1)
        self.mean = nn.Parameter(torch.zeros(norm_params_dim), requires_grad=False)  # TODO: Should not appear in state_dict.
        self.std_dev = nn.Parameter(torch.ones(norm_params_dim), requires_grad=False)
        # self.dropout = hparams.dropout
        self.batch_first = hparams.batch_first
        self.batch_dim = 0 if hparams.batch_first else 1
        self.time_dim = 1 if hparams.batch_first else 0

        # Create hparams for pre-net.
        self.hparams_prenet = copy.deepcopy(hparams)
        self.hparams_prenet.model_type = hparams.pre_net_model_type
        self.hparams_prenet.model_name = hparams.pre_net_model_name
        self.hparams_prenet.model_path = hparams.pre_net_model_path
        # Remove embedding functions when they should not been passed.
        if not hparams.pass_embs_to_pre_net:
            self.hparams_prenet.f_get_emb_index = None

        # Create pre-net from type if not None, or try to load it by given path, or default path plus name.
        from idiaptts.src.neural_networks.pytorch.ModelHandlerPyTorch import ModelHandlerPyTorch
        self.model_handler_prenet = ModelHandlerPyTorch()
        if self.hparams_prenet.model_type is not None:
            prenet_dim_in = (dim_in[0] - (0 if not hparams.f_get_emb_index or hparams.pass_embs_to_pre_net
                                          else len(hparams.f_get_emb_index)),
                             *dim_in[1:])
            self.model_handler_prenet.create_model(self.hparams_prenet,
                                                   prenet_dim_in,
                                                   dim_out)
        elif self.hparams_prenet.model_path is not None:
            self.model_handler_prenet.model, *_ = self.model_handler_prenet.load_model(self.hparams_prenet.model_path,
                                                                                       self.hparams_prenet,
                                                                                       verbose=True)
        elif self.hparams_prenet.model_name is not None:
            self.hparams_prenet.model_path = os.path.join(self.hparams_prenet.out_dir,
                                                          self.hparams_prenet.networks_dir,
                                                          self.hparams_prenet.model_name)
            self.model_handler_prenet.model, *_ = self.model_handler_prenet.load_model(self.hparams_prenet.model_path,
                                                                                       self.hparams_prenet,
                                                                                       verbose=True)
        else:
            self.logger.warning("No pre-net specified.")

        if self.model_handler_prenet.model is not None:
            self.model_handler_prenet.model.save_intermediate_outputs = True  # Used by RNNDyn.
            self.add_module("pre_net", self.model_handler_prenet.model)  # Properly register parameters of submodule.
            if not hparams.train_pre_net:
                for param in self.model_handler_prenet.model.parameters():
                    param.requires_grad = False

        self.prenet_group_index_of_alpha = -2
        self.embedding_dim = hparams.speaker_emb_dim
        if hparams.num_speakers is None:
            self.logger.warning("Number of speaker is not defined. Assume only one speaker for embedding.")
            self.num_speakers = 1
        else:
            self.num_speakers = hparams.num_speakers
        self.pass_embs_to_pre_net = hparams.pass_embs_to_pre_net

        self.n = hparams.num_coded_sps
        self.alpha_range = 0.2
        self.has_deltas = hparams.add_deltas
        self.max_polynomial = 2 * self.n

        # Reuse pre-net embeddings or create new ones if non exist yet.
        if not hparams.pass_embs_to_pre_net or not self.model_handler_prenet.model:
            self.embeddings = nn.Embedding(self.num_speakers, self.embedding_dim)
        else:
            self.embeddings = self.model_handler_prenet.model.emb_groups[0]

        # Attach alpha layer to selected pre-net layer.
        if self.model_handler_prenet.model is not None:
            pre_net_layer_group = self.model_handler_prenet.model.layer_groups[self.prenet_group_index_of_alpha]
            self.alpha_layer = nn.Linear(pre_net_layer_group.out_dim * (2 if pre_net_layer_group.is_rnn else 1)
                                         + self.embedding_dim, 1)
        else:
            self.alpha_layer = nn.Linear(np.prod(dim_in) + self.embedding_dim, 1)

        # self.alpha_layer = nn.Linear(53, 1)
        # self.alpha_layer = nn.Linear(self.embedding_dim, 1)

        # self.all_pass_warp_matrix = None
        # self.precision = 100
        # self.eps = 1e-45  # float(np.finfo(np.float32).eps)
        # self.pre_compute_warp_matrices(self.precision, requires_recursive_grad=True)

        self.computation_dtype = 'torch.FloatTensor'  # torch.float32 cannot be pickled.
        self.w_matrix_3d = self.gen_w_matrix_3d()

        # self.index_vec_pos = torch.arange(0, 2 * self.n, dtype=self.computation_dtype)
        # index_vec_neg_sign = torch.tensor([v * pow(-1, i) for i, v in enumerate(range(0, 2 * self.n))],
        #                                   dtype=self.computation_dtype, requires_grad=False).sign()
        # index_vec_neg_sign[0] = 1.
        #
        # self.w_matrix_3d_sign = self.w_matrix_3d.sign().type(self.computation_dtype)
        # self.w_matrix_3d_sign = torch.stack((self.w_matrix_3d_sign, self.w_matrix_3d_sign * index_vec_neg_sign[None, None, :]))
        # self.w_matrix_3d_log = torch.log(self.w_matrix_3d.abs()).type(self.index_vec_pos.dtype)

        self.w_matrix_3d = self.w_matrix_3d.type(self.computation_dtype)

        # self.compare_with_recursive(self.alpha_range)
        if self.use_gpu:
            self.w_matrix_3d = self.w_matrix_3d.cuda()
            # self.w_matrix_3d_sign = self.w_matrix_3d_sign.cuda()
            # self.w_matrix_3d_log = self.w_matrix_3d_log.cuda()
            # self.index_vec_pos = self.index_vec_pos.cuda()

    def set_norm_params(self, mean, std_dev):
        mean = torch.from_numpy(mean) if isinstance(mean, np.ndarray) else mean
        std_dev = torch.from_numpy(std_dev) if isinstance(std_dev, np.ndarray) else std_dev
        mean = mean.type(torch.float32)
        std_dev = std_dev.type(torch.float32)

        if self.use_gpu:
            mean = mean.cuda()
            std_dev = std_dev.cuda()

        self.mean = torch.nn.Parameter(mean, requires_grad=False)
        self.std_dev = torch.nn.Parameter(std_dev, requires_grad=False)

    def init_hidden(self, batch_size=1):
        return self.model_handler_prenet.model.init_hidden(batch_size)

    def set_gpu_flag(self, use_gpu):
        self.use_gpu = use_gpu
        if self.use_gpu:
            # self.alpha_list = self.alpha_list.cuda(async=True)  # Lazy loading.
            # self.warp_matrix_list = self.warp_matrix_list.cuda()  # Always required, no lazy loading.
            self.w_matrix_3d = self.w_matrix_3d.cuda()
            if self.mean is not None:
                self.mean = self.mean.cuda()
            if self.std_dev is not None:
                self.std_dev = self.std_dev.cuda()
        self.model_handler_prenet.model.set_gpu_flag(use_gpu)

    # def log_per_batch(self):
    #     emb = self.embeddings(torch.zeros(1).to(self.embeddings.weight.device).long()).unsqueeze(0)
    #
    #     alpha = self.alpha_layer(emb)
    #     alpha = torch.tanh(alpha) * self.alpha_range
    #     logging.info("alpha={}".format(alpha[0][0]))

    # def log_per_test(self):
    #     self.log_per_batch()

    def gen_w_matrix_3d(self):
        """
        Computes the entries with the formula for m-th row and k-th column:
        A(m, k) = 1/(k-1)! * sum_{n=max(0, k-m}}^k (k choose n) * (m+n-1)! / (m+n-k)! * (-1)^{n+k+m} alpha^{2n+m-k}
        The entries are stored as a vector corresponding to the polynomials of alpha (1, alpha, alpha^2,..., alpha^{M-1}).

        :return:     Values in warping matrix.
        """
        grad_matrix = np.zeros((self.n, self.n, self.max_polynomial), dtype=np.float64) #(np.float64 if self.n >= 32 else np.float32))  # 32! = 2.6E35

        grad_matrix[0, 0, 0] = 1.0
        max_degree = 0
        for m in range(0, self.n):
            for k in range(1, self.n):
                k_fac = 1 / math.factorial(k-1) if k > 0 else 1
                for n in range(max(0, k-m), k + 1):
                    w = ncr(k, n) * math.pow(-1, n + m + k) # * (2 * n + m - k)
                    w_fac = math.factorial(m + n - 1) if m + n - 1 > 0 else 1
                    w_fac /= math.factorial(m + n - k) if m + n - k > 0 else 1
                    w *= w_fac * k_fac
                    # if w != 0.0:
                    degree = 2 * n + m - k # - 1
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
        grad_matrix = torch.from_numpy(grad_matrix)

        grad_matrix = torch.transpose(grad_matrix, 0, 1).contiguous()
        return grad_matrix

    def gen_warp_matrix_recursively(self, alpha, requires_recursive_grad=True):
        m = [[torch.empty((1,), dtype=alpha.dtype, requires_grad=requires_recursive_grad) for x in range(self.n)] for y in range(self.n)]
        n = self.n
        m[0][0] = torch.ones((1,), dtype=alpha.dtype, requires_grad=requires_recursive_grad)
        for r in range(1, n):
            m[r][0] = m[r - 1][0] * alpha
        for c in range(1, n):
            m[0][c] = torch.zeros((1,), dtype=alpha.dtype, requires_grad=requires_recursive_grad)  # Fix for transpose.
            for r in range(1, n):
                m[r][c] = m[r - 1][c - 1] + alpha * (m[r - 1][c] - m[r][c - 1])

        return torch.cat([torch.cat(x) for x in m]).view(self.n, self.n)

    # def compare_with_recursive(self, alpha_range, precision=0.05, delta=0.001):
    #     """
    #     Compare the element-wise computed gradient matrix with the recursively generate matrix for alphas in
    #     range(-alpha_range, alpha_range, precision).
    #
    #     :param alpha_range:           Range of alpha to test in.
    #     :param precision:             Precision used for steps in that range.
    #     :param delta:                 Allowed delta of error.
    #
    #     :return:
    #     """
    #     assert(precision < 2 * alpha_range)  # Precision must fit in range.
    #
    #     for alpha_value in np.arange(-alpha_range, alpha_range + precision, precision):
    #         # Alpha value which receives the final gradient.
    #         alpha = torch.tensor(alpha_value, dtype=self.w_matrix_3d.dtype, requires_grad=True)
    #         alpha_eps = alpha
    #         alpha_eps = alpha_eps.repeat([1000, 1])
    #
    #         # Compute the warp matrix for each alpha.
    #         warp_matrix = self.get_warp_matrix_log(alpha_eps)
    #
    #         # Create the reference matrix recursively for the given alpha.
    #         ref_matrix = self.gen_warp_matrix_recursively(alpha)
    #
    #         # Compute the error.
    #         dist = (warp_matrix[10] - ref_matrix).abs()
    #         max_error = (dist / (ref_matrix.abs() + 1e-6)).max()
    #         error = dist.sum()
    #
    #         err_msg = "Max error between w_matrix_3d and recursive reference is {:.5f}% for alpha={:.2f}.".format(
    #             max_error * 100, alpha_value)
    #         logging.error(err_msg)
    #         if max_error > delta:
    #             raise ValueError(err_msg)
    #
    #         # Compute the gradient ratio error.
    #         ref_matrix.sum().backward()
    #         real_grad = torch.tensor(alpha.grad)
    #         alpha.grad.zero_()
    #         warp_matrix.sum().backward()
    #         approx_grad = alpha.grad / len(alpha_eps)
    #         dist_grad = (real_grad - approx_grad).abs()
    #         error_ratio = (dist_grad / real_grad.abs())
    #
    #         err_msg = "Gradient error between w_matrix_3d and recursive reference is {:.5f}% for alpha={:.2f}.".format(
    #             error_ratio * 100., alpha_value)
    #         logging.error(err_msg)
    #         if error_ratio > delta:
    #             raise ValueError(err_msg)
    #
    #     return True

    def pre_compute_warp_matrices(self, precision, requires_recursive_grad=True):
        """
        Recursively pre-compute warping matrices for [-1, 1] with given precision.
        Unpractical because recursive backwards pass takes too long.
        """

        self.warp_matrix_list = list()
        self.alpha_list = list()

        for alpha in range(-1 * precision, 1 * precision, 1):
            alpha = torch.tensor(alpha, dtype=torch.float32, requires_grad=requires_recursive_grad)
            self.alpha_list.append(alpha.unsqueeze(0) + self.precision)
            self.warp_matrix_list.append(self.gen_warp_matrix_recursively(alpha / precision, requires_recursive_grad).unsqueeze(0))  # Create "continuous" matrix.

        self.warp_matrix_list = torch.cat(self.warp_matrix_list)
        self.alpha_list = torch.cat(self.alpha_list)

        if not requires_recursive_grad:
            self.warp_matrix_list.requires_grad_(True)
            self.alpha_list.requires_grad_(True)

        if self.use_gpu:
            self.alpha_list = self.alpha_list.cuda(async=True)  # Lazy loading.
            self.warp_matrix_list = self.warp_matrix_list.cuda()  # Always required, no lazy loading.

    def get_warp_matrix_index(self, alphas):
        """
        Compute warping matrix for vector of alphas in log space.

        :param alphas:       Vector of alphas with time and batch dimension merged (TB x 1).
        :return:             Warping matrix for each alpha value with merged time and batch dimension (TB x n x n).
        """

        alphas = (alphas + 1.) * self.precision

        alpha0 = alphas.floor().detach().squeeze(-1)
        alpha1 = alpha0 + 1

        warp_matrix0 = self.warp_matrix_list[alpha0.long()]
        warp_matrix1 = self.warp_matrix_list[alpha1.long()]

        W0 = (alpha1.unsqueeze(-1) - alphas).unsqueeze(-1).expand_as(warp_matrix0)
        W1 = (alphas - alpha0.unsqueeze(-1)).unsqueeze(-1).expand_as(warp_matrix1)

        warp_matrix = W0 * warp_matrix0 + W1 * warp_matrix1  # Doesn't work with negative indices.
        warp_matrix = warp_matrix.view(-1, self.n, self.n)  # Merge time and batch dimension to use torch.bmm().

        return warp_matrix

    def get_warp_matrix_log(self, alphas):
        """
        Compute warping matrix for vector of alphas in log space.

        :param alphas:       Vector of alphas with time and batch dimension merged (TB x 1).
        :return:             Warping matrix for each alpha value with merged time and batch dimension (TB x n x n).
        """

        # Compute log of alpha^{0..2*self.n} with alpha==0 save.
        # alphas[alphas == 0] = alphas[alphas == 0] + self.eps
        log_alpha = alphas.abs().log()
        alpha_vec = torch.mm(log_alpha, self.index_vec_pos.view(1, -1))  # TB x 2N

        # Compute elements of sum of warping matrix in third dimension.
        w_matrix_3d_expanded = self.w_matrix_3d_log.expand(alpha_vec.shape[0], *self.w_matrix_3d_log.shape)  # TB x n x n x 2n
        w_matrix_3d_alpha = w_matrix_3d_expanded + alpha_vec[:, None, None, :]
        w_matrix_3d_alpha = w_matrix_3d_alpha.exp()

        # Apply the correct sign to the elements in third dimension.
        alpha_positive_indices = alphas[:, 0] < 0
        w_matrix_3d_alpha = torch.index_select(self.w_matrix_3d_sign, dim=0, index=alpha_positive_indices.long()) * w_matrix_3d_alpha

        # Compute actual warping matrix.
        warp_matrix = w_matrix_3d_alpha.sum(dim=3)

        return warp_matrix

    def get_warp_matrix(self, alphas):
        """
        Compute warping matrix for vector of alphas.

        :param alphas:       Vector of alphas with time and batch dimension merged (TB x 1).
        :return:             Warping matrix for each alpha value with merged time and batch dimension (TB x n x n).
        """

        # Create alpha polynomial vector.
        alpha_list = [torch.ones((alphas.shape), dtype=self.w_matrix_3d.dtype, device=alphas.device, requires_grad=True)]
        for i in range(1, self.max_polynomial):
            alpha_list.append(alpha_list[i - 1] * alphas)
        alpha_vec = torch.cat(alpha_list, dim=1).unsqueeze(-1)  # T x 2n x 1

        # Do a batched matrix multiplication to get the elements of the warp matrix.
        grad_matrix_flat = self.w_matrix_3d.view(self.n * self.n, self.w_matrix_3d.shape[-1])  # n x n x 2n -> n^2 x 2n
        grad_matrix_ext_flat = grad_matrix_flat.expand(alpha_vec.shape[0], *grad_matrix_flat.shape[0:])  # TB x n^2 x 2n
        warp_matrix = torch.matmul(grad_matrix_ext_flat, alpha_vec).view(-1, self.n, self.n)  # TB x n x n

        return warp_matrix

    def forward_sample(self, in_tensor, alphas=None):
        """Forward one tensor through the layer."""
        if isinstance(in_tensor, np.ndarray):
            in_tensor = torch.from_numpy(in_tensor)
        in_tensor = in_tensor[:, None].to(self.w_matrix_3d.device)

        if alphas is not None:
            if isinstance(alphas, np.ndarray):
                alphas = torch.from_numpy(alphas)
            alphas = alphas[:, None].to(self.w_matrix_3d.device)

        return self.forward(in_tensor,
                            hidden=None,
                            seq_length_input=(len(in_tensor),),
                            max_length_input=(len(in_tensor),),
                            alphas=alphas)

    def forward(self, inputs, hidden, seq_length_input, max_length_input, target=None, seq_lengths_output=None, alphas=None):

        batch_size = inputs.shape[self.batch_dim]
        # num_frames = inputs.shape[self.time_dim]

        # Code for testing fixed alphas.
        if alphas is not None:
            alphas = alphas.to(self.w_matrix_3d.device)
            inputs = inputs.to(self.w_matrix_3d.device)
            output = inputs.type(self.w_matrix_3d.dtype)
            group_output = inputs.type(self.w_matrix_3d.dtype)
        else:
            inputs_emb = inputs[:, :, -1]
            if not self.pass_embs_to_pre_net:
                inputs = inputs[:, :, :-1]
            output, hidden = self.model_handler_prenet.model(inputs, hidden, seq_length_input, max_length_input, target, seq_lengths_output)
            group_output = self.model_handler_prenet.model.layer_groups[self.prenet_group_index_of_alpha].output

            group_output = group_output.view(output.shape[self.time_dim], batch_size, -1) # View operation to get rid of possible bidirectional outputs.

            emb = self.embeddings(inputs_emb.long()) #[None, ...]  # Use speaker 0 for everything for now.
            #emb = emb.expand(-1, group_output.shape[1], -1) if self.batch_first else emb.expand(group_output.shape[0], -1, -1)  # Expand the temporal dimension.
            emb_out = torch.cat((emb, group_output), dim=2)

            alphas = self.alpha_layer(emb_out)
            # alphas = self.alpha_layer(inputs[:, :, 86:347:5])
            alphas = torch.tanh(alphas) * self.alpha_range
            # alphas = torch.zeros((*output.shape[:2], 1), device=output.device)

        alphas = alphas.view(-1, 1).type(self.w_matrix_3d.dtype)  # Merge time and batch dimension.
        warp_matrix = self.get_warp_matrix(alphas)

        if self.has_deltas:
            warped_feature_list = list()
            for start_index in range(0, 3):
                feature = output[:, :, start_index*self.n:(start_index + 1)*self.n]  # Select spectral features.

                # Denormalize before warping.
                if self.std_dev is not None:
                    feature = feature * self.std_dev[start_index*self.n:(start_index + 1)*self.n]
                if self.mean is not None:
                    feature = feature + self.mean[start_index*self.n:(start_index + 1)*self.n]
                feature[:, :, 0::self.n] /= 2.  # Adaptation for single-sided spectrogram.

                # Merge time and batch axis, do batched vector matrix multiplication with a (1 x N * N x N) matrix
                # multiplication, split time and batch axis back again.
                feature_warped = torch.bmm(feature.view(-1, 1, *feature.shape[2:]), warp_matrix).view(-1, batch_size, *feature.shape[2:])

                feature_warped[:, :, 0::self.n] *= 2.  # Adaptation for single-sided spectrogram.
                # Normalize again for further processing.
                if self.mean is not None:
                    feature_warped = feature_warped - self.mean[start_index * self.n:(start_index + 1) * self.n]
                if self.std_dev is not None:
                    feature_warped = feature_warped / self.std_dev[start_index*self.n:(start_index + 1)*self.n]

                warped_feature_list.append(feature_warped)
            output = torch.cat((*warped_feature_list, output[:, :, 3*self.n:]), dim=2)
        else:
            feature = output[:, :, :self.n]  # Select spectral features.

            # Denormalize before warping.
            if self.std_dev is not None:
                feature = feature * self.std_dev
            if self.mean is not None:
                feature = feature + self.mean
            feature[:, :, 0] /= 2.  # Adaptation for single-sided spectrogram.

            # Merge time and batch axis, do batched vector matrix multiplication with a (1 x N * N x N) matrix
            # multiplication, split time and batch axis back again.
            feature_warped = torch.bmm(feature.view(-1, 1, *feature.shape[2:]), warp_matrix).squeeze(1).view(-1, batch_size, *feature.shape[2:])

            feature_warped[:, :, 0] *= 2.  # Adaptation for single-sided spectrogram.
            # Normalize again for further processing.
            if self.mean is not None:
                feature_warped = feature_warped - self.mean
            if self.std_dev is not None:
                feature_warped = feature_warped / self.std_dev

            output = torch.cat((feature_warped, output[:, :, self.n:]), dim=2)

        return output, (hidden, alphas.view(-1, batch_size))


def main():
    from idiaptts.src.model_trainers.vtln.VTLNSpeakerAdaptionModelTrainer import VTLNSpeakerAdaptionModelTrainer
    hparams = VTLNSpeakerAdaptionModelTrainer.create_hparams()
    hparams.use_gpu = False
    hparams.voice = "English"
    hparams.model_name = "WarpingLayerTest.nn"
    hparams.add_deltas = True
    hparams.num_coded_sps = 30
    # hparams.num_questions = 505
    hparams.num_questions = 425
    hparams.out_dir = "experiments/" + hparams.voice + "/VTLNArtificiallyWarped/"
    hparams.data_dir = os.path.realpath("database")
    hparams.model_name = "warping_layer_test"
    hparams.synth_dir = hparams.out_dir
    batch_size = 2
    dir_world_labels = os.path.join("experiments", hparams.voice, "WORLD")

    from idiaptts.src.data_preparation.world.WorldFeatLabelGen import WorldFeatLabelGen
    gen_in = WorldFeatLabelGen(dir_world_labels, add_deltas=hparams.add_deltas, num_coded_sps=hparams.num_coded_sps)
    gen_in.get_normalisation_params(gen_in.dir_labels)

    from idiaptts.src.model_trainers.AcousticModelTrainer import AcousticModelTrainer
    trainer = AcousticModelTrainer("experiments/" + hparams.voice + "/WORLD", "experiments/" + hparams.voice + "/questions", "ignored", hparams.num_questions, hparams)

    sp_mean = gen_in.norm_params[0][:hparams.num_coded_sps * (3 if hparams.add_deltas else 1)]
    sp_std_dev = gen_in.norm_params[1][:hparams.num_coded_sps * (3 if hparams.add_deltas else 1)]
    wl = WarpingLayer((hparams.num_coded_sps,), (hparams.num_coded_sps,), hparams)
    wl.set_norm_params(sp_mean, sp_std_dev)

    # id_list = ["dorian/doriangray_16_00199"]
    id_list = ["p225/p225_051"]
    hparams.num_speakers = 1

    t_benchmark = 0
    for id_name in id_list:
        for idx, alpha in enumerate(np.arange(-0.15, 0.2, 0.05)):
            out_dir = hparams.out_dir + "alpha_{0:0.2f}/".format(alpha)
            makedirs_safe(out_dir)

            sample = WorldFeatLabelGen.load_sample(id_name, os.path.join("experiments", hparams.voice, "WORLD"),
                                                   add_deltas=True, num_coded_sps=hparams.num_coded_sps)
            sample_pre = gen_in.preprocess_sample(sample)
            coded_sps = sample_pre[:, :hparams.num_coded_sps * (3 if hparams.add_deltas else 1)]

            alpha_vec = np.ones((coded_sps.shape[0], 1)) * alpha

            coded_sps = coded_sps[:len(alpha_vec), None, ...].repeat(batch_size, 1)  # Copy data in batch dimension.
            alpha_vec = alpha_vec[:, None, None].repeat(batch_size, 1)  # Copy data in batch dimension.

            t_start = timer()
            mfcc_warped, (_, nn_alpha) = wl(torch.from_numpy(coded_sps), None,
                                            (len(coded_sps),), (len(coded_sps),),
                                            alphas=torch.from_numpy(alpha_vec))
            mfcc_warped.sum().backward()
            t_benchmark += timer() - t_start
            assert((mfcc_warped[:, 0] == mfcc_warped[:, 1]).all())  # Compare results for cloned coded_sps within batch.
            if alpha == 0:
                assert((mfcc_warped == coded_sps).all())  # Compare results for no warping.
            sample_pre[:len(mfcc_warped), :hparams.num_coded_sps * (3 if hparams.add_deltas else 1)] = mfcc_warped[:, 0].detach()

            sample_post = gen_in.postprocess_sample(sample_pre)
            # Manually create samples without normalisation but with deltas.
            sample_pre = (sample_pre * gen_in.norm_params[1] + gen_in.norm_params[0]).astype(np.float32)

            if np.isnan(sample_pre).any():
                raise ValueError("Detected nan values in output features for {}.".format(id_name))
            # Save warped features.
            makedirs_safe(os.path.dirname(os.path.join(out_dir, id_name)))
            sample_pre.tofile(os.path.join(out_dir, id_name + WorldFeatLabelGen.ext_deltas))

            hparams.synth_dir = out_dir
            Synthesiser.run_world_synth({id_name: sample_post}, hparams)

    print("Process time for {} runs: {}".format(len(id_list) * idx, timedelta(seconds=t_benchmark)))


if __name__ == "__main__":
    main()
