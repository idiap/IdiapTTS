#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#

"""Module description.

"""

# System imports.
import os
import logging
import math
import copy
from functools import reduce
from typing import Union, Any, List, Optional, cast, Dict, Tuple
from timeit import default_timer as timer
from datetime import timedelta

# Third-party imports.
import numpy as np
import torch
import torch.nn as nn

# Local source tree imports.
from idiaptts.misc.utils import ncr, makedirs_safe
from idiaptts.src.Synthesiser import Synthesiser
from idiaptts.src.neural_networks.EmbeddingConfig import EmbeddingConfig
from idiaptts.src.neural_networks.pytorch.layers.AllPassWarp import AllPassWarp
from idiaptts.src.neural_networks.pytorch.models.ModelConfig import ModelConfig
from idiaptts.src.neural_networks.pytorch.GradientScaling import grad_scaling


class AllPassWarpLayer(nn.Module):
    logger = logging.getLogger(__name__)

    class Config:
        def __init__(self,
                     alpha_layer_in_dims: List[int],
                     alpha_ranges: List[float],
                     batch_first: bool,
                     warp_matrix_size: int,
                     gradient_scaling: float = None,
                     mean: torch.Tensor = None,
                     n_frames_per_step: int = 1,
                     std_dev: torch.Tensor = None,
                     **kwargs):
            if alpha_layer_in_dims and alpha_ranges is not None:
                assert len(alpha_layer_in_dims) == len(alpha_ranges), \
                    "Number of alpha_layer_dims ({}) has to match alpha_ranges ({}).".format(
                        len(alpha_layer_in_dims, len(alpha_ranges)))
            assert warp_matrix_size > 0, "warp_matrix_size must be greater than 0."

            self.alpha_layer_dims = alpha_layer_in_dims
            self.alpha_ranges = alpha_ranges
            self.batch_first = batch_first
            self.warp_matrix_size = warp_matrix_size
            self.gradient_scaling = gradient_scaling
            self.n_frames_per_step = n_frames_per_step

            if mean is not None:
                if isinstance(mean, np.ndarray):
                    mean = torch.from_numpy(mean)
                elif not isinstance(mean, torch.Tensor):
                    raise TypeError("mean has to be of type numpy.ndarray or torch.Tensor.")
                mean = mean.float()
            if std_dev is not None:
                if isinstance(std_dev, np.ndarray):
                    std_dev = torch.from_numpy(std_dev)
                elif not isinstance(std_dev, torch.Tensor):
                    raise TypeError("std_dev has to be of type numpy.ndarray or torch.Tensor.")
                std_dev = std_dev.float()
            self.mean = mean
            self.std_dev = std_dev

        def create_model(self):
            return AllPassWarpLayer(self)

        # def __getstate__(self):
        #     state = self.__dict__.copy()
        #     state.pop('mean', None)
        #     state.pop('std_dev', None)
        #     return state

        # def __setstate__(self, state):
        #     self.__dict__.update(state)

    def __init__(self, config: Config):
        super().__init__()

        # Store parameters.
        self.dim_in = config.alpha_layer_dims
        self.warp_matrix_size = config.warp_matrix_size
        self.n_frames_per_step = config.n_frames_per_step
        self.gradient_scaling = config.gradient_scaling

        # norm_params_size = self.warp_matrix_size * (3 if hparams.add_deltas else 1)
        self.register_buffer("mean", config.mean if hasattr(config, "mean") else None)
        self.register_buffer("std_dev", config.std_dev if hasattr(config, "std_dev") else None)
        self.batch_first = config.batch_first
        self.batch_dim = 0 if config.batch_first else 1
        self.time_dim = 1 if config.batch_first else 0

        if config.alpha_layer_dims is not None:
            self.alpha_layers = self._setup_alpha_layers(config.alpha_layer_dims)
        self.alpha_ranges = config.alpha_ranges

        self.all_pass_warp = AllPassWarp(config.warp_matrix_size)

    def _setup_alpha_layers(self, alpha_layer_dims):
        alpha_layers = list()
        for warping_layer_idx, layer_dim in enumerate(alpha_layer_dims):
            new_layer = nn.Linear(layer_dim, self.n_frames_per_step)
            # torch.nn.init.xavier_uniform_(new_layer.weight, gain=torch.nn.init.calculate_gain("tanh"))
            # torch.nn.init.xavier_normal_(new_layer.weight, gain=torch.nn.init.calculate_gain("tanh"))
            alpha_layers.append(new_layer)

        return nn.ModuleList(alpha_layers)

    def init_hidden(self, batch_size=1):
        return None

    def forward_sample(self, in_tensor, alphas):
        """Forward one tensor through the layer."""
        if isinstance(in_tensor, np.ndarray):
            in_tensor = torch.from_numpy(in_tensor)
        in_tensor = in_tensor.unsqueeze(0 if self.batch_first else 1).float().to(self.all_pass_warp.w_matrix_3d.device)

        if type(alphas) not in [list, tuple]:
            alphas = (alphas,)

        alphas = [torch.from_numpy(alpha) if isinstance(alpha, np.ndarray) else alpha for alpha in alphas]
        alphas = [alpha.unsqueeze(0 if self.batch_first else 1)
                       .type(self.all_pass_warp.w_matrix_3d.dtype)
                       .to(self.all_pass_warp.w_matrix_3d.device) for alpha in alphas]

        return self.forward_fixed_alphas(in_tensor, alphas=alphas)

    def forward_fixed_alphas(self, input_, alphas):

        assert alphas is not None, "This forward call requires alphas."

        input_ = self._denormalise(input_)
        # output = torch.empty(input_.shape, dtype=input_.dtype, device=input_.device, requires_grad=False)
        output, combined_alphas = self.all_pass_warp(input_, alphas)
        output = self._normalise(output)

        return output, combined_alphas

    def forward(self, inputs, lengths, max_lengths, **kwargs) -> Tuple[List[torch.Tensor], Dict]:
        inputs, *alpha_layers_inputs = inputs
        alphas = self.get_alphas(*alpha_layers_inputs)
        return [*self.forward_fixed_alphas(inputs, alphas), *alphas], {"lengths": lengths, "max_lengths": max_lengths}

    def get_alphas(self, *alpha_layer_inputs):
        alphas = list()
        for idx in range(len(self.alpha_layers)):
            alphas.append(self.get_alpha(alpha_layer_inputs[idx], idx))

        return alphas

    def get_alpha(self, alpha_layers_input, alpha_layer_idx):

        alphas = self.alpha_layers[alpha_layer_idx](alpha_layers_input)
        scaled_alphas = torch.tanh(alphas) * self.alpha_ranges[alpha_layer_idx]
        if self.gradient_scaling is not None:
            scaled_alphas = grad_scaling.apply(scaled_alphas, self.gradient_scaling)

        B = scaled_alphas.shape[self.batch_dim]
        T = scaled_alphas.shape[self.time_dim]
        if self.batch_first:
            scaled_alphas = scaled_alphas.view(B, T * self.n_frames_per_step, 1)
        else:
            scaled_alphas = scaled_alphas.transpose(0, 1).contiguous().view(B, T * self.n_frames_per_step, 1).transpose(0, 1)
        return scaled_alphas

    def set_norm_params(self, mean, std_dev):
        mean = torch.from_numpy(mean) if isinstance(mean, np.ndarray) else mean
        std_dev = torch.from_numpy(std_dev) if isinstance(std_dev, np.ndarray) else std_dev
        current_device = next(self.parameters()).device
        self.mean = mean.type(torch.float32).to(current_device)
        self.std_dev = std_dev.type(torch.float32).to(current_device)

    def _normalise(self, features):
        if self.mean is not None:
            features = features - self.mean
        if self.std_dev is not None:
            features = features / self.std_dev
        return features

    def _denormalise(self, features):
        # if self.mean is None:
        #     self.logger.warning("Mean seems to be None is that intended?")
        if self.std_dev is not None:
            features = features * self.std_dev
        if self.mean is not None:
            features = features + self.mean
        return features


def main():
    from idiaptts.src.model_trainers.vtln.VTLNSpeakerAdaptionModelTrainer import VTLNSpeakerAdaptionModelTrainer
    hparams = VTLNSpeakerAdaptionModelTrainer.create_hparams()
    hparams.use_gpu = False
    hparams.voice = "full"
    # hparams.model_name = "AllPassWarpModelTest"
    hparams.add_deltas = False
    hparams.num_coded_sps = 30
    hparams.synth_fs = 16000
    # hparams.num_questions = 505
    hparams.num_questions = 425
    dir_experiments = os.path.join("experiments{}".format(hparams.synth_fs), hparams.voice)
    hparams.out_dir = os.path.join(dir_experiments, "VTLNArtificiallyWarped")
    hparams.data_dir = os.path.realpath("database")
    hparams.model_name = "all_pass_warp_test"
    hparams.synth_dir = hparams.out_dir
    batch_size = 2
    dir_world_labels = os.path.join(dir_experiments, "WORLD")

    # hparams.add_hparam("warp_matrix_size", hparams.num_coded_sps)
    # hparams.alpha_ranges = [0.2, ]

    from idiaptts.src.data_preparation.world.WorldFeatLabelGen import WorldFeatLabelGen
    gen_in = WorldFeatLabelGen(dir_world_labels,
                               add_deltas=hparams.add_deltas,
                               num_coded_sps=hparams.num_coded_sps,
                               num_bap=hparams.num_bap)
    gen_in.get_normalisation_params(gen_in.dir_labels)
    sp_mean = gen_in.norm_params[0][:hparams.num_coded_sps * (3 if hparams.add_deltas else 1)]
    sp_std_dev = gen_in.norm_params[1][:hparams.num_coded_sps * (3 if hparams.add_deltas else 1)]
    all_pass_warp_model = AllPassWarpLayer.Config(None, None, batch_first=hparams.batch_first,
                                                  warp_matrix_size=hparams.num_coded_sps, mean=sp_mean,
                                                  std_dev=sp_std_dev).create_model()
    all_pass_warp_model.set_norm_params(sp_mean, sp_std_dev)

    from idiaptts.src.model_trainers.AcousticModelTrainer import AcousticModelTrainer
    trainer = AcousticModelTrainer(**AcousticModelTrainer.legacy_support_init(dir_experiments + "/WORLD",
                                                                              dir_experiments + "/questions",
                                                                              "ignored",
                                                                              hparams.num_questions,
                                                                              hparams))

    id_list = ["DC/DC_a01"]

    t_benchmark = 0
    for id_name in id_list:
        sample = WorldFeatLabelGen.load_sample(id_name,
                                               os.path.join(dir_experiments, "WORLD"),
                                               add_deltas=hparams.add_deltas,
                                               num_coded_sps=hparams.num_coded_sps,
                                               num_bap=hparams.num_bap,
                                               sp_type=hparams.sp_type)
        sample_pre = gen_in.preprocess_sample(sample)
        coded_sps = sample_pre[:, :hparams.num_coded_sps * (3 if hparams.add_deltas else 1)].copy()
        coded_sps = coded_sps[:, None, ...].repeat(batch_size, 1)  # Copy data in batch dimension.

        for idx, alpha in enumerate(np.arange(-0.2, 0.21, 0.05)):
            out_dir = os.path.join(hparams.out_dir, "alpha_{0:0.2f}".format(alpha))
            makedirs_safe(out_dir)

            alpha_vec = np.ones((coded_sps.shape[0], 1)) * alpha
            alpha_vec = alpha_vec[:, None].repeat(batch_size, 1)  # Copy data in batch dimension.

            t_start = timer()
            sp_warped, *_ = all_pass_warp_model.forward_fixed_alphas(
                    torch.from_numpy(coded_sps.copy()),
                    alphas=torch.tensor(alpha_vec, requires_grad=True).to(
                        all_pass_warp_model.all_pass_warp.w_matrix_3d.device).type(
                            all_pass_warp_model.all_pass_warp.w_matrix_3d.dtype))
            sp_warped.sum().backward()
            t_benchmark += timer() - t_start
            # assert((mfcc_warped[:, 0] == mfcc_warped[:, 1]).all())  # Compare results for cloned coded_sps within batch.
            if np.isclose(alpha, 0):
                assert np.isclose(sp_warped.detach().cpu().numpy(), coded_sps).all()  # Compare no warping results.
            sample_pre[:len(sp_warped), :hparams.num_coded_sps * (3 if hparams.add_deltas else 1)] = sp_warped[:, 0].detach()

            sample_post = gen_in.postprocess_sample(sample_pre, apply_mlpg=False)
            # Manually create samples without normalisation but with deltas.
            sample_pre_with_deltas = (sample_pre * gen_in.norm_params[1] + gen_in.norm_params[0]).astype(np.float32)

            if np.isnan(sample_pre_with_deltas).any():
                raise ValueError("Detected nan values in output features for {}.".format(id_name))
            # Save warped features.
            makedirs_safe(os.path.dirname(os.path.join(out_dir, id_name)))
            sample_pre_with_deltas.tofile(os.path.join(out_dir, id_name + "." + WorldFeatLabelGen.ext_deltas))

            hparams.synth_dir = out_dir
            # sample_no_deltas = WorldFeatLabelGen.convert_from_world_features(*WorldFeatLabelGen.convert_to_world_features(sample, contains_deltas=hparams.add_deltas, num_coded_sps=hparams.num_coded_sps, num_bap=hparams.num_bap))
            Synthesiser.run_world_synth({id_name: sample_post}, hparams)

    print("Process time for {} runs: {}, average: {}".format(len(id_list) * idx, timedelta(seconds=t_benchmark), timedelta(seconds=t_benchmark) / (len(id_list) * idx)))


if __name__ == "__main__":
    main()
