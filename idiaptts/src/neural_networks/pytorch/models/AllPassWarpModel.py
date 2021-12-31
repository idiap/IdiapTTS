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

import numpy as np
from timeit import default_timer as timer
from datetime import timedelta

# Third-party imports.
import torch
import torch.nn as nn

# Local source tree imports.
from idiaptts.misc.utils import ncr, makedirs_safe
from idiaptts.src.Synthesiser import Synthesiser
from idiaptts.src.neural_networks.EmbeddingConfig import EmbeddingConfig
from idiaptts.src.neural_networks.pytorch.layers.AllPassWarp import AllPassWarp


class AllPassWarpModel(nn.Module):
    IDENTIFIER = "AllPassWarp"
    logger = logging.getLogger(__name__)

    def __init__(self, dim_in, dim_out, hparams):
        super().__init__()

        # Store parameters.
        self.use_gpu = hparams.use_gpu
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.warp_matrix_size = hparams.warp_matrix_size if hasattr(hparams, "warp_matrix_size")\
                                                         else hparams.num_coded_sps
        self.has_deltas = hparams.add_deltas
        self.prenet_group_index_of_alpha = -2 if not hparams.has_value("prenet_group_index_of_alpha") else hparams.prenet_group_index_of_alpha
        self.n_frames_per_step = hparams.n_frames_per_step if hparams.has_value("n_frames_per_step") else 1

        norm_params_size = self.warp_matrix_size * (3 if hparams.add_deltas else 1)
        self.register_buffer("mean", torch.zeros(norm_params_size))
        self.register_buffer("std_dev", torch.ones(norm_params_size))
        self.batch_first = hparams.batch_first
        self.batch_dim = 0 if hparams.batch_first else 1
        self.time_dim = 1 if hparams.batch_first else 0

        self.pre_net = None
        self.alpha_layers = None
        self.alpha_ranges = None

        self._verify_embeddings(hparams)
        self.embeddings_configs = hparams.embeddings if hparams.has_value("embeddings") else list()
        self._setup_pre_net((dim_in, dim_out), hparams)
        self._setup_embeddings(hparams)
        self._setup_alpha_layers(hparams)
        self.all_pass_warp = AllPassWarp(self.warp_matrix_size, hparams)

        self.set_gpu_flag(hparams.use_gpu)

    def _verify_embeddings(self, hparams):
        if not hasattr(hparams, "embeddings"):
            hparams.add_hparam("embeddings", None)
            return
        if hparams.embeddings is None:
            return

        # For a single embedding hparams.embeddings can be a single tuple of (f_get_emb_index, pass_to_pre_net, ...).
        if hparams.embeddings[0] is EmbeddingConfig:
            hparams.embeddings = [hparams.embeddings, ]

        for idx, (emb_config, pass_to_pre_net, pass_to_warping_layer_list) in enumerate(hparams.embeddings):
            assert callable(emb_config.f_get_emb_index), "Embedding function {} is not callable.".format(idx)
            assert type(pass_to_pre_net) is bool, "pass_to_pre_net of embedding {} is not a bool".format(idx)
            # pass_to_warping_layer can be a single bool which should hold for all warping layers or
            # a list which should be the same length as the number of warping layers.
            if type(pass_to_warping_layer_list) is bool:
                assert hasattr(hparams, "alpha_ranges") and len(hparams.alpha_ranges) > 0,\
                    "Embedding {} should be passed to all warping layers but no warping layers exist.".format(idx)
            else:
                assert len(pass_to_warping_layer_list) == len(hparams.alpha_ranges),\
                    "pass_to_warping_layer in embedding {} has {} elements but doesn't match {} warping layers."\
                    .format(idx, len(pass_to_warping_layer_list), len(hparams.alpha_ranges))
                assert all(type(pass_to_warping_layer) is bool for pass_to_warping_layer in pass_to_warping_layer_list),\
                       "All elements of pass_to_warping_layer in embedding {} must be of type bool.".format(idx)

    def _setup_pre_net(self, dim_in_out, hparams):

        if not hparams.pre_net_model_name and not hparams.pre_net_model_path and not hparams.pre_net_model_type:
            self.logger.warning("No pre-net specified.")
            return

        hparams_prenet = AllPassWarpModel._create_pre_net_hparams(hparams)

        if self._load_pre_net_flag(hparams):
            self._load_pre_net(hparams_prenet)
        else:
            self._create_pre_net(dim_in_out, hparams_prenet)

        self.add_module("pre_net", self.pre_net)  # Properly register parameters of submodule.
        self.pre_net_requires_grad(hparams.train_pre_net)

    @staticmethod
    def _create_pre_net_hparams(hparams):
        hparams_prenet = copy.deepcopy(hparams)
        hparams_prenet.model_type = hparams.pre_net_model_type
        hparams_prenet.model_name = hparams.pre_net_model_name
        hparams_prenet.model_path = hparams.pre_net_model_path
        hparams_prenet.add_hparam("save_intermediate_outputs", True)
        hparams_prenet.del_hparam("embeddings")

        f_get_emb_indices = list()
        emb_configs = list()
        if hparams.has_value("embeddings"):
            for emb_config, pass_to_pre_net, _ in hparams.embeddings:
                if pass_to_pre_net:
                    f_get_emb_indices.append(emb_config.f_get_emb_index)
                    emb_configs.append(emb_config)
            hparams_prenet.setattr_no_type_check("f_get_emb_index", f_get_emb_indices)
            hparams_prenet.add_hparam("embeddings", emb_configs)

        return hparams_prenet

    def _load_pre_net_flag(self, hparams):
        if hparams.load_from_checkpoint:
            # If the whole model is loaded it includes the pre-net, however, a new pre-net can only be created if its
            # type is known. Thus if the type is unknown we still need to load it.
            return hparams.pre_net_model_type is None
        if not hasattr(hparams, "load_pre_net_from_checkpoint"):
            return False
        return hparams.load_pre_net_from_checkpoint

    def _load_pre_net(self, hparams):
        from idiaptts.src.neural_networks.pytorch.ModelHandlerPyTorch import ModelHandlerPyTorch
        from idiaptts.src.model_trainers.ModelTrainer import ModelTrainer

        model_path = ModelTrainer.get_model_path(hparams)
        self.pre_net, *_ = ModelHandlerPyTorch.load_model(model_path, hparams, verbose=True)

    def _create_pre_net(self, dim_in_out, hparams):
        from idiaptts.src.neural_networks.pytorch.ModelFactory import ModelFactory
        self.pre_net = ModelFactory.create(hparams.model_type, *dim_in_out, hparams)

    def pre_net_requires_grad(self, requires_grad):
        for param in self.pre_net.parameters():
            param.requires_grad = requires_grad

    def _setup_embeddings(self, hparams):
        self.embeddings = nn.ModuleList()

        if hparams.embeddings is None:
            return

        if self.pre_net is not None:
            pre_net_embeddings = self.pre_net.get_embeddings()
            pre_net_embs_idx = 0

        for idx, (emb_config, pass_to_pre_net, pass_to_warping_layer_list) in enumerate(hparams.embeddings):
            if pass_to_pre_net:
                assert self.pre_net is not None,\
                    "Embedding {} should be passed to pre-net but no pre-net exists.".format(idx)
                assert len(pre_net_embeddings) > pre_net_embs_idx,\
                    "Embedding {} should be passed to pre-net but pre-net expects only {} embeddings."\
                        .format(idx, len(pre_net_embeddings))
                assert pre_net_embeddings[str(pre_net_embs_idx)].num_embeddings == emb_config.num_embeddings
                assert pre_net_embeddings[str(pre_net_embs_idx)].embedding_dim == emb_config.embedding_dim
                self.embeddings.append(pre_net_embeddings[str(pre_net_embs_idx)])
                pre_net_embs_idx += 1
            else:
                self.embeddings.append(nn.Embedding(emb_config.num_embeddings,
                                                    emb_config.embedding_dim,
                                                    **emb_config.args))

    def _setup_alpha_layers(self, hparams):
        self.alpha_ranges = list()

        assert hparams.has_value("alpha_ranges"), "At least one alpha range has to be given."
        alpha_layers = list()

        for warping_layer_idx, alpha_range in enumerate(hparams.alpha_ranges):
            cumulated_embedding_dim = 0

            if hparams.embeddings is not None:
                for idx, (emb_config, pass_to_pre_net, pass_to_warping_layer_list)in enumerate(hparams.embeddings):
                    if type(pass_to_warping_layer_list) is bool:
                        pass_to_warping_layer = pass_to_warping_layer_list
                    else:
                        pass_to_warping_layer = pass_to_warping_layer_list[warping_layer_idx]

                    if pass_to_warping_layer:
                        cumulated_embedding_dim += emb_config.embedding_dim

            if self.pre_net is not None:
                out_dims = self.pre_net.get_group_out_dim(self.prenet_group_index_of_alpha)
                pre_net_out_dim = sum(out_dims) if type(out_dims) in [list, tuple] else out_dims
            else:
                pre_net_out_dim = np.prod(self.dim_in)

            new_layer = nn.Linear(pre_net_out_dim + cumulated_embedding_dim, self.n_frames_per_step)
            alpha_layers.append(new_layer)
            self.alpha_ranges.append(alpha_range)

        self.alpha_layers = nn.ModuleList(alpha_layers)

    def init_hidden(self, batch_size=1):
        hiddens = list()
        if self.pre_net is not None:
            hiddens.append(self.pre_net.init_hidden(batch_size))
        return hiddens

    def set_gpu_flag(self, use_gpu):
        self.use_gpu = use_gpu
        self.all_pass_warp.set_gpu_flag(use_gpu)

        if self.pre_net is not None:
            self.pre_net.set_gpu_flag(use_gpu)

    def forward_sample(self, in_tensor, alphas=None):
        """Forward one tensor through the layer."""
        if isinstance(in_tensor, np.ndarray):
            in_tensor = torch.from_numpy(in_tensor)
        in_tensor = in_tensor[:, None].to(self.all_pass_warp.w_matrix_3d.device)

        if alphas is not None:
            if isinstance(alphas, np.ndarray):
                alphas = torch.from_numpy(alphas)
            alphas = alphas[:, None].to(self.all_pass_warp.w_matrix_3d.device)

        if alphas is None:
            return self.forward(in_tensor,
                                hidden=None,
                                seq_length_input=(len(in_tensor),),
                                max_length_input=(len(in_tensor),))
        else:
            return self.forward_fixed_alphas(in_tensor,
                                             hidden=None,
                                             alphas=alphas)

    def forward_fixed_alphas(self, inputs, hidden, alphas):

        assert alphas is not None, "This forward call requires alphas."

        # Code for testing fixed alphas.
        alphas = alphas.to(self.all_pass_warp.w_matrix_3d.device)
        alphas = alphas.type(self.all_pass_warp.w_matrix_3d.dtype)
        inputs = inputs.to(self.all_pass_warp.w_matrix_3d.device)
        output = inputs.type(self.all_pass_warp.w_matrix_3d.dtype)

        spectral_features_dim = self.warp_matrix_size * (3 if self.has_deltas else 1)
        spectral_features = output[:, :, :spectral_features_dim]
        spectral_features = self._denormalise(spectral_features)

        new_output = torch.empty(output.shape, dtype=output.dtype, device=output.device, requires_grad=False)
        self.all_pass_warp(spectral_features, alphas, out_tensor=new_output)

        new_output[:, :, 0:spectral_features_dim] = self._normalise(new_output[:, :, 0:spectral_features_dim])
        new_output[:, :, spectral_features_dim:] = output[:, :, spectral_features_dim:]

        return new_output, (hidden, alphas)

    def forward(self, inputs, hidden, seq_length_input, max_length_input, target=None, seq_lengths_output=None, *extra_labels):
        if self.pre_net is not None:
            output, hidden, pre_net_intermediate_output = self.get_pre_net_output(inputs,
                                                                                  hidden,
                                                                                  seq_length_input,
                                                                                  max_length_input,
                                                                                  target,
                                                                                  seq_lengths_output,
                                                                                  *extra_labels)
        else:
            if len(self.embeddings) > 0:
                output = inputs[:, :, :-len(self.embeddings)]
            else:
                output = inputs
            pre_net_intermediate_output = output
        alphas = self.get_alphas(inputs, pre_net_intermediate_output)

        spectral_features_dim = self.warp_matrix_size * (3 if self.has_deltas else 1)
        spectral_features = output[:, :, :spectral_features_dim]
        spectral_features = self._denormalise(spectral_features)

        new_output = torch.empty(output.shape, dtype=output.dtype, device=output.device, requires_grad=False)
        self.all_pass_warp(spectral_features, alphas, out_tensor=new_output)

        new_output[:, :, 0:spectral_features_dim] = self._normalise(new_output[:, :, 0:spectral_features_dim])
        new_output[:, :, spectral_features_dim:] = output[:, :, spectral_features_dim:]

        return new_output, (hidden, alphas)

    def get_pre_net_output(self, inputs, hidden, seq_length_input, max_length_input, target, seq_lengths_output, *extra_labels):
        batch_size = inputs.shape[self.batch_dim]
        inputs_pre_net = self._remove_non_pre_net_embeddings(inputs)
        pre_net_output, hidden = self.pre_net(inputs_pre_net, hidden, seq_length_input, max_length_input, target, seq_lengths_output, *extra_labels)

        pre_net_intermediate_output = self.pre_net.get_intermediate_output(self.prenet_group_index_of_alpha)
        # View operation to get rid of possible bidirectional outputs.
        # pre_net_intermediate_output = pre_net_intermediate_output.view(pre_net_intermediate_output.shape[self.time_dim],
        #                                                                batch_size,
        #                                                                -1)

        return pre_net_output, hidden, pre_net_intermediate_output

    def _remove_non_pre_net_embeddings(self, inputs):
        if len(self.embeddings) > 0:
            inputs_without_embeddings = inputs[:, :, :-len(self.embeddings)]
            embeddings_to_pass = [inputs[:, :, -len(self.embeddings) + emb_idx].unsqueeze(2)
                                  for emb_idx, (emb_config, pass_to_pre_net, *_) in enumerate(self.embeddings_configs)
                                  if pass_to_pre_net]
            return torch.cat((inputs_without_embeddings, *embeddings_to_pass), dim=2)
        else:
            return inputs

    def get_alphas(self, inputs, pre_net_intermediate_output):
        alphas = list()
        for idx in range(len(self.alpha_layers)):
            alphas.append(self.get_alpha(inputs, pre_net_intermediate_output, idx))

        return alphas

    def get_alpha(self, inputs, pre_net_intermediate_output, alpha_layer_idx):

        embs = self._get_alpha_layer_embeddings(inputs, alpha_layer_idx)

        # if len(embs) == 0:
        #     embs_and_output = pre_net_intermediate_output
        # else:
        embs = self._match_embs_sequence_length(embs, pre_net_intermediate_output)
        embs_and_output = torch.cat((*embs, pre_net_intermediate_output), dim=2)

        alphas = self.alpha_layers[alpha_layer_idx](embs_and_output)
        scaled_alphas = torch.tanh(alphas) * self.alpha_ranges[alpha_layer_idx]

        B = scaled_alphas.shape[self.batch_dim]
        if self.batch_first:
            scaled_alphas = alphas.view(B, -1, 1)
        else:
            scaled_alphas = scaled_alphas.transpose(0, 1).contiguous().view(B, -1, 1).transpose(0, 1)
        return scaled_alphas

    def _match_embs_sequence_length(self, embs, output):
        desired_length = output.shape[self.time_dim]
        repeats = (desired_length if self.time_dim == 0 else 1, desired_length if self.time_dim == 1 else 1, 1)
        output_embs = []
        for embedding in embs:
            if embedding.shape[self.time_dim] != desired_length:
                embedding = embedding[0:1] if self.time_dim == 0 else embedding[:, 0:1]
                embedding = embedding.repeat(repeats)
            output_embs.append(embedding)
        return output_embs

    def _get_alpha_layer_embeddings(self, inputs, alpha_layer_idx):
        return [self.embeddings[idx](inputs[:, :, -len(self.embeddings) + idx].long())
                for idx, (emb_config, _, pass_to_warping_layers_list) in enumerate(self.embeddings_configs)
                    if (pass_to_warping_layers_list if type(pass_to_warping_layers_list) is bool
                                                    else pass_to_warping_layers_list[alpha_layer_idx])]

    def set_norm_params(self, mean, std_dev):
        mean = torch.from_numpy(mean) if isinstance(mean, np.ndarray) else mean
        std_dev = torch.from_numpy(std_dev) if isinstance(std_dev, np.ndarray) else std_dev
        mean = mean.type(torch.float32)
        std_dev = std_dev.type(torch.float32)

        if self.use_gpu:
            mean = mean.cuda()
            std_dev = std_dev.cuda()

        self.mean = mean
        self.std_dev = std_dev

    def _normalise(self, features):
        if self.mean is not None:
            features = features - self.mean
        if self.std_dev is not None:
            features = features / self.std_dev
        return features

    def _denormalise(self, features):
        if self.std_dev is not None:
            features = features * self.std_dev
        if self.mean is not None:
            features = features + self.mean
        return features

    def get_embeddings(self):
        return self.embeddings


def main():
    from idiaptts.src.model_trainers.vtln.VTLNSpeakerAdaptionModelTrainer import VTLNSpeakerAdaptionModelTrainer
    hparams = VTLNSpeakerAdaptionModelTrainer.create_hparams()
    hparams.use_gpu = False
    hparams.voice = "English"
    hparams.model_name = "AllPassWarpModelTest.nn"
    hparams.add_deltas = True
    hparams.num_coded_sps = 30
    # hparams.num_questions = 505
    hparams.num_questions = 425
    hparams.out_dir = os.path.join("experiments", hparams.voice, "VTLNArtificiallyWarped")
    hparams.data_dir = os.path.realpath("database")
    hparams.model_name = "all_pass_warp_test"
    hparams.synth_dir = hparams.out_dir
    batch_size = 2
    dir_world_labels = os.path.join("experiments", hparams.voice, "WORLD")

    # hparams.add_hparam("warp_matrix_size", hparams.num_coded_sps)
    hparams.alpha_ranges = [0.2, ]

    from idiaptts.src.data_preparation.world.WorldFeatLabelGen import WorldFeatLabelGen
    gen_in = WorldFeatLabelGen(dir_world_labels,
                               add_deltas=hparams.add_deltas,
                               num_coded_sps=hparams.num_coded_sps,
                               num_bap=hparams.num_bap)
    gen_in.get_normalisation_params(gen_in.dir_labels)

    from idiaptts.src.model_trainers.AcousticModelTrainer import AcousticModelTrainer
    trainer = AcousticModelTrainer("experiments/" + hparams.voice + "/WORLD", "experiments/" + hparams.voice + "/questions", "ignored", hparams.num_questions, hparams)

    sp_mean = gen_in.norm_params[0][:hparams.num_coded_sps * (3 if hparams.add_deltas else 1)]
    sp_std_dev = gen_in.norm_params[1][:hparams.num_coded_sps * (3 if hparams.add_deltas else 1)]
    all_pass_warp_model = AllPassWarpModel((hparams.num_coded_sps,), (hparams.num_coded_sps,), hparams)
    all_pass_warp_model.set_norm_params(sp_mean, sp_std_dev)

    # id_list = ["dorian/doriangray_16_00199"]
    # id_list = ["p225/p225_051", "p277/p277_012", "p278/p278_012", "p279/p279_012"]
    id_list = ["p225/p225_051"]

    t_benchmark = 0
    for id_name in id_list:
        sample = WorldFeatLabelGen.load_sample(id_name,
                                               os.path.join("experiments", hparams.voice, "WORLD"),
                                               add_deltas=True,
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
            sp_warped, (_, nn_alpha) = all_pass_warp_model(torch.from_numpy(coded_sps.copy()),
                                                           None,
                                                           (len(coded_sps),),
                                                           (len(coded_sps),),
                                                           alphas=torch.tensor(alpha_vec, requires_grad=True))
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
