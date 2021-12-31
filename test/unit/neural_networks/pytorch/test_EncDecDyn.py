#
# Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#


import unittest

import copy
import os
import torch
import jsonpickle

from idiaptts.misc.utils import makedirs_safe
import idiaptts.src.neural_networks.pytorch.models.rnn_dyn as rnn_dyn
import idiaptts.src.neural_networks.pytorch.models.enc_dec_dyn as enc_dec_dyn


class TestEncDecDyn(unittest.TestCase):

    out_dir = None

    @classmethod
    def setUpClass(cls):
        cls.out_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                   type(cls()).__name__)
        makedirs_safe(cls.out_dir)  # Create class name directory.

    @classmethod
    def tearDownClass(cls):
        os.rmdir(cls.out_dir)  # Remove class name directory, should be empty.

    def _get_encoder_config(self, out_dim):
        return enc_dec_dyn.Config.ModuleConfig(
            name="Encoder",
            config=rnn_dyn.Config(
                in_dim=1,
                layer_configs=[
                    rnn_dyn.Config.LayerConfig(
                        layer_type="Embedding", num_embeddings=2,
                        embedding_dim=4),
                    rnn_dyn.Config.LayerConfig(
                        layer_type="Conv1d", out_dim=6, kernel_size=3,
                        nonlin="ReLU", padding=1),
                    rnn_dyn.Config.LayerConfig(layer_type="BatchNorm1d"),
                    rnn_dyn.Config.LayerConfig(
                        layer_type="Linear", out_dim=out_dim)],
            ),
            input_names=["phonemes"],
            output_names=["phoneme_embeddings"],
            process_group=0)

    def _get_encoder_vae_pool_last_config(self, in_dim, out_dim):
        return enc_dec_dyn.Config.ModuleConfig(
            name="EncoderVAE",
            config=rnn_dyn.Config(
                in_dim=in_dim,
                layer_configs=[
                    rnn_dyn.Config.LayerConfig(
                        layer_type="Conv1d", out_dim=2, num_layers=1,
                        kernel_size=3, stride=2, padding=1),
                    rnn_dyn.Config.LayerConfig(
                        layer_type="Conv1d", out_dim=4, num_layers=2,
                        kernel_size=3, stride=2, padding=1),
                    rnn_dyn.Config.LayerConfig(layer_type="GRU", out_dim=8),
                    rnn_dyn.Config.LayerConfig(layer_type="PoolLast",
                                               batch_first=True),
                    rnn_dyn.Config.LayerConfig(layer_type="VAE",
                                               out_dim=out_dim)]),
            input_names=["acoustic_features"],
            output_names=["emb_z", "emb_mu", "emb_logvar"],
            process_group=0)

    def _get_encoder_vae_config(self, in_dim, out_dim):
        return enc_dec_dyn.Config.ModuleConfig(
            name="EncoderVAE",
            config=rnn_dyn.Config(
                in_dim=in_dim,
                layer_configs=[
                    rnn_dyn.Config.LayerConfig(
                        layer_type="Conv1d", out_dim=2, num_layers=1,
                        kernel_size=3, padding=1),
                    rnn_dyn.Config.LayerConfig(
                        layer_type="Conv1d", out_dim=4, num_layers=2,
                        kernel_size=3, padding=1),
                    rnn_dyn.Config.LayerConfig(layer_type="GRU", out_dim=8),
                    rnn_dyn.Config.LayerConfig(layer_type="VAE",
                                               out_dim=out_dim)]),
            input_names=["acoustic_features"],
            output_names=["emb_z", "emb_mu", "emb_logvar"],
            process_group=0)

    def _get_encoder_embedding_config(self, out_dim):
        return enc_dec_dyn.Config.ModuleConfig(
            name="Embedding",
            config=rnn_dyn.Config(
                in_dim=1,
                layer_configs=[
                    rnn_dyn.Config.LayerConfig(layer_type='Embedding',
                                               num_embeddings=2,
                                               embedding_dim=out_dim)
                ]),
            input_names=['emb_idx'],
            output_names=["emb_z"],
            process_group=0
        )

    def _get_fixed_attention_config(self):
        return enc_dec_dyn.Config.DecoderConfig(
            attention_args={enc_dec_dyn.ATTENTION_GROUND_TRUTH: "attention_matrix"},
            attention_config=enc_dec_dyn.FIXED_ATTENTION,
            input_names="phoneme_embeddings",
            name="FixedAttention",
            output_names="upsampled_phoneme_embeddings",
            process_group=1,
            n_frames_per_step=1
        )

    def _get_fixed_attention_decoder_config(self, audio_encoder_dim, in_dim,
                                            out_dim, n_frames_per_step,
                                            p_teacher_forcing):
        return enc_dec_dyn.Config.DecoderConfig(
            attention_args={enc_dec_dyn.ATTENTION_GROUND_TRUTH: "attention_matrix"},
            attention_config=enc_dec_dyn.FIXED_ATTENTION,
            teacher_forcing_input_names=["acoustic_features"],
            config=rnn_dyn.Config(
                in_dim=in_dim,
                layer_configs=[
                    rnn_dyn.Config.LayerConfig(
                        layer_type="FC", out_dim=8, nonlin="RELU"),
                    rnn_dyn.Config.LayerConfig(layer_type="LSTM", out_dim=4)
                ]
            ),
            input_names=["phoneme_embeddings", "emb_z"],
            name="Decoder",
            n_frames_per_step=n_frames_per_step,
            p_teacher_forcing=p_teacher_forcing,
            pre_net_config=rnn_dyn.Config(
                in_dim=out_dim,
                layer_configs=[
                    rnn_dyn.Config.LayerConfig(
                        layer_type="linear", out_dim=audio_encoder_dim,
                        nonlin="relu", num_layers=2)
                ]
            ),
            process_group=1,
            projection_configs=[
                enc_dec_dyn.Config.ProjectionConfig(
                    config=rnn_dyn.Config(
                        in_dim=4,
                        layer_configs=[rnn_dyn.Config.LayerConfig(
                            layer_type="FC", out_dim=out_dim * n_frames_per_step)]),
                    name="AcousticFeaturesProjector",
                    output_names=["pred_intermediate_acoustic_features"],
                    out_dim=out_dim,
                    is_autoregressive_input=True
                )
            ]
        )

    def _get_parallel_decoder_config(self, in_dim, out_dim):
        return enc_dec_dyn.Config.ModuleConfig(
            config=rnn_dyn.Config(
                in_dim=in_dim,
                layer_configs=[
                    rnn_dyn.Config.LayerConfig(layer_type="FC", out_dim=8,
                                               nonlin="RELU", dropout=0.1),
                    rnn_dyn.Config.LayerConfig(layer_type="LSTM",
                                               out_dim=out_dim, dropout=0.1)
                ]
            ),
            input_names=["upsampled_phoneme_embeddings", "emb_z"],
            name="ParallelDecoder",
            process_group=2,
            output_names=["pred_acoustic_features"]
        )

    def _get_postnet_config(self, out_dim):
        return enc_dec_dyn.Config.ModuleConfig(
            name="Postnet",
            config=rnn_dyn.Config(
                in_dim=out_dim,
                layer_configs=[rnn_dyn.Config.LayerConfig(layer_type="Conv1d",
                                                          out_dim=4,
                                                          kernel_size=3),
                               rnn_dyn.Config.LayerConfig(
                                    layer_type="BatchNorm1d"),
                               rnn_dyn.Config.LayerConfig(layer_type="ReLU"),
                               rnn_dyn.Config.LayerConfig(layer_type="Linear",
                                                          out_dim=out_dim)]
            ),
            input_names=["pred_intermediate_acoustic_features"],
            output_names=["pred_acoustic_features"],
            process_group=2
        )

    def test_fixed_attention_auto_regressive_b1(self):
        encoder_dim = 12
        vae_dim = 6
        audio_encoder_dim = 6
        n_frames_per_step = 5
        p_teacher_forcing = 0.5
        decoder_dim = 15

        model_config = enc_dec_dyn.Config(
            modules=[
                self._get_encoder_config(encoder_dim),
                self._get_encoder_vae_pool_last_config(decoder_dim, vae_dim),
                self._get_fixed_attention_decoder_config(
                    audio_encoder_dim, encoder_dim + audio_encoder_dim + vae_dim,
                    decoder_dim, n_frames_per_step, p_teacher_forcing),
                self._get_postnet_config(decoder_dim)
            ]
        )
        model = model_config.create_model()

        phoneme_seq_length = torch.tensor((10,), dtype=torch.long)
        phoneme_max_length = torch.tensor(10, dtype=torch.long)
        seq_length = torch.tensor((100,), dtype=torch.long)
        max_length = torch.tensor(100, dtype=torch.long)
        batch_size = 1

        test_input = {}
        test_input["phonemes"] = torch.ones(
            [batch_size, phoneme_seq_length.max(), 1]).long()
        test_input["acoustic_features"] = torch.ones(
            [batch_size, seq_length.max(), decoder_dim])
        test_input["attention_matrix"] = torch.zeros(
            (batch_size, max_length, phoneme_max_length))
        seq_length_dict = {"phonemes": phoneme_seq_length,
                           "acoustic_features": seq_length}
        max_length_dict = {"phonemes": phoneme_max_length,
                           "acoustic_features": max_length}

        model.init_hidden(batch_size)
        output = model(test_input, seq_length_dict, max_length_dict)

        self.assertEqual(torch.Size([batch_size, seq_length.max(), decoder_dim]),
                         output["pred_acoustic_features"].shape)

    def test_fixed_attention_auto_regressive(self):
        encoder_dim = 12
        vae_dim = 6
        audio_encoder_dim = 6
        n_frames_per_step = 5
        p_teacher_forcing = 0.5
        decoder_dim = 15

        model_config = enc_dec_dyn.Config(
            modules=[
                self._get_encoder_config(encoder_dim),
                self._get_encoder_vae_pool_last_config(decoder_dim, vae_dim),
                self._get_fixed_attention_decoder_config(
                    audio_encoder_dim, encoder_dim + audio_encoder_dim + vae_dim,
                    decoder_dim, n_frames_per_step, p_teacher_forcing),
                self._get_postnet_config(decoder_dim)
            ]
        )
        model = model_config.create_model()

        phoneme_seq_length = torch.tensor((10, 12), dtype=torch.long)
        phoneme_max_length = torch.tensor(12, dtype=torch.long)
        seq_length = torch.tensor((100, 75), dtype=torch.long)
        max_length = torch.tensor(100, dtype=torch.long)
        batch_size = 2

        test_input = {}
        test_input["phonemes"] = torch.ones(
            [batch_size, phoneme_seq_length.max(), 1]).long()
        test_input["acoustic_features"] = torch.ones(
            [batch_size, seq_length.max(), decoder_dim])
        test_input["attention_matrix"] = torch.zeros(
            (batch_size, max_length, phoneme_max_length))
        seq_length_dict = {"phonemes": phoneme_seq_length,
                           "acoustic_features": seq_length}
        max_length_dict = {"phonemes": phoneme_max_length,
                           "acoustic_features": max_length}

        model.init_hidden(batch_size)
        output = model(test_input, seq_length_dict, max_length_dict)

        self.assertEqual(torch.Size([batch_size,
                                     phoneme_seq_length.max(),
                                     encoder_dim]),
                         output["phoneme_embeddings"].shape)
        self.assertEqual(torch.Size([batch_size, 1, vae_dim]),
                         output["emb_mu"].shape)
        self.assertEqual(torch.Size(
            [batch_size,
             seq_length.max() // n_frames_per_step, phoneme_seq_length.max()]),
            output["attention"].shape)
        self.assertEqual(torch.Size(
            [batch_size, seq_length.max(), decoder_dim]),
            output["pred_intermediate_acoustic_features"].shape)
        self.assertEqual(torch.Size(
            [batch_size, seq_length.max(), decoder_dim]),
            output["pred_acoustic_features"].shape)
        self.assertTrue(
            (phoneme_seq_length == seq_length_dict["phoneme_embeddings"]).any())
        self.assertTrue(
            (phoneme_max_length == max_length_dict["phoneme_embeddings"]).any())
        self.assertTrue(
            (torch.tensor((1, 1), dtype=torch.long) == max_length_dict["emb_mu"]).any())
        self.assertTrue(
            (seq_length == seq_length_dict["pred_intermediate_acoustic_features"]).any())
        self.assertTrue(
            (seq_length == seq_length_dict["pred_acoustic_features"]).any())

        expected_params = 1 + 2 * 3  # Phoneme encoder: 1 Emb + 3 Conv (weight & bias)
        expected_params += 2 * 3 + 4 + 1  # Acoustic encoder: 3 Conv + GRU + VAE projection
        expected_params += 2 + 4  # Decoder: Linear + LSTM
        expected_params += 2 * 2  # Decoder pre-net: 2 Linear
        expected_params += 2  # Decoder projection: 1 Linear
        expected_params += 3 * 2  # Postnet: 1 Conv + 1 BatchNorm1d + 1 Linear
        self.assertEqual(expected_params, len([*model.named_parameters()]))
        output["pred_acoustic_features"].sum().backward()

    def test_fixed_attention_batched_b1(self):
        encoder_dim = 12
        vae_dim = 6
        audio_encoder_dim = 6
        n_frames_per_step = 5
        p_teacher_forcing = 1
        decoder_dim = 15

        model_config = enc_dec_dyn.Config(
            modules=[
                self._get_encoder_config(encoder_dim),
                self._get_encoder_vae_pool_last_config(decoder_dim, vae_dim),
                self._get_fixed_attention_decoder_config(
                    audio_encoder_dim, encoder_dim + audio_encoder_dim + vae_dim,
                    decoder_dim, n_frames_per_step, p_teacher_forcing),
                self._get_postnet_config(decoder_dim)
            ]
        )
        model = model_config.create_model()

        phoneme_seq_length = torch.tensor((10,), dtype=torch.long)
        phoneme_max_length = torch.tensor(10, dtype=torch.long)
        seq_length = torch.tensor((100,), dtype=torch.long)
        max_length = torch.tensor(100, dtype=torch.long)
        batch_size = 1

        test_input = {}
        test_input["phonemes"] = torch.ones(
            [batch_size, phoneme_seq_length.max(), 1]).long()
        test_input["acoustic_features"] = torch.ones(
            [batch_size, seq_length.max(), decoder_dim])
        test_input["attention_matrix"] = torch.zeros(
            (batch_size, max_length, phoneme_max_length))
        seq_length_dict = {"phonemes": phoneme_seq_length,
                           "acoustic_features": seq_length,
                           "attention_matrix": seq_length}
        max_length_dict = {"phonemes": phoneme_max_length,
                           "acoustic_features": max_length,
                           "attention_matrix": max_length}

        model.init_hidden(batch_size)
        output = model(test_input, seq_length_dict, max_length_dict)

        self.assertEqual(torch.Size([batch_size, seq_length.max(), decoder_dim]),
                         output["pred_acoustic_features"].shape)

    def test_fixed_attention_batched(self):
        encoder_dim = 12
        vae_dim = 6
        audio_encoder_dim = 6
        n_frames_per_step = 5
        p_teacher_forcing = 1
        decoder_dim = 15

        model_config = enc_dec_dyn.Config(
            modules=[
                self._get_encoder_config(encoder_dim),
                self._get_encoder_vae_pool_last_config(decoder_dim, vae_dim),
                self._get_fixed_attention_decoder_config(
                    audio_encoder_dim, encoder_dim + audio_encoder_dim + vae_dim,
                    decoder_dim, n_frames_per_step, p_teacher_forcing),
                self._get_postnet_config(decoder_dim)
            ]
        )
        model = model_config.create_model()

        phoneme_seq_length = torch.tensor((10, 12), dtype=torch.long)
        phoneme_max_length = torch.tensor(12, dtype=torch.long)
        seq_length = torch.tensor((100, 75), dtype=torch.long)
        max_length = torch.tensor(100, dtype=torch.long)
        batch_size = 2

        test_input = {}
        test_input["phonemes"] = torch.ones(
            [batch_size, phoneme_seq_length.max(), 1]).long()
        test_input["acoustic_features"] = torch.ones(
            [batch_size, seq_length.max(), decoder_dim])
        test_input["attention_matrix"] = torch.zeros(
            (batch_size, max_length, phoneme_max_length))
        seq_length_dict = {"phonemes": phoneme_seq_length,
                           "acoustic_features": seq_length,
                           "attention_matrix": seq_length}
        max_length_dict = {"phonemes": phoneme_max_length,
                           "acoustic_features": max_length,
                           "attention_matrix": max_length}

        model.init_hidden(batch_size)
        output = model(test_input, seq_length_dict, max_length_dict)

        self.assertEqual(
            torch.Size([batch_size, phoneme_seq_length.max(), encoder_dim]),
            output["phoneme_embeddings"].shape)
        self.assertEqual(torch.Size([batch_size, 1, vae_dim]),
                         output["emb_mu"].shape)
        self.assertEqual(torch.Size([batch_size, seq_length.max(), decoder_dim]),
                         output["pred_intermediate_acoustic_features"].shape)
        self.assertEqual(torch.Size([batch_size, seq_length.max(), decoder_dim]),
                         output["pred_acoustic_features"].shape)
        self.assertTrue(
            (phoneme_seq_length == seq_length_dict["phoneme_embeddings"]).any())
        self.assertTrue(
            (phoneme_max_length == max_length_dict["phoneme_embeddings"]).any())
        self.assertTrue(
            (torch.tensor((1, 1), dtype=torch.long) == max_length_dict["emb_mu"]).any())
        self.assertTrue(
            (seq_length == seq_length_dict["pred_intermediate_acoustic_features"]).any())
        self.assertTrue(
            (seq_length == seq_length_dict["pred_acoustic_features"]).any())

        expected_params = 1 + 2 * 3  # Phoneme encoder: 1 Emb + 3 Conv (weight & bias)
        expected_params += 2 * 3 + 4 + 1  # Acoustic encoder: 3 Conv + GRU + VAE projection
        expected_params += 2 + 4  # Decoder: Linear + LSTM
        expected_params += 2 * 2  # Decoder pre-net: 2 Linear
        expected_params += 2  # Decoder projection: 1 Linear
        expected_params += 3 * 2  # Postnet: 1 Conv + 1 BatchNorm1d + 1 Linear
        self.assertEqual(expected_params, len([*model.named_parameters()]))
        output["pred_acoustic_features"].sum().backward()

    def test_fixed_attention_parallel_decoder_b1(self):
        encoder_dim = 12
        vae_dim = 6
        audio_encoder_dim = 6
        n_frames_per_step = 5
        p_teacher_forcing = 1
        decoder_dim = 15

        model_config = enc_dec_dyn.Config(
            modules=[
                self._get_encoder_config(encoder_dim),
                self._get_encoder_vae_config(decoder_dim, vae_dim),
                self._get_fixed_attention_config(),
                self._get_parallel_decoder_config(encoder_dim + vae_dim,
                                                  decoder_dim)
            ]
        )
        model = model_config.create_model()

        phoneme_seq_length = torch.tensor((10,), dtype=torch.long)
        phoneme_max_length = torch.tensor(10, dtype=torch.long)
        seq_length = torch.tensor((100,), dtype=torch.long)
        max_length = torch.tensor(100, dtype=torch.long)
        batch_size = 1

        test_input = {}
        test_input["phonemes"] = torch.ones(
            [batch_size, phoneme_seq_length.max(), 1]).long()
        test_input["acoustic_features"] = torch.ones(
            [batch_size, seq_length.max(), decoder_dim])
        test_input["attention_matrix"] = torch.zeros(
            (batch_size, max_length, phoneme_max_length))
        seq_length_dict = {"phonemes": phoneme_seq_length,
                           "acoustic_features": seq_length,
                           "attention_matrix": seq_length}
        max_length_dict = {"phonemes": phoneme_max_length,
                           "acoustic_features": max_length,
                           "attention_matrix": max_length}

        model.init_hidden(batch_size)
        output = model(test_input, seq_length_dict, max_length_dict)

        self.assertEqual(torch.Size([batch_size, seq_length.max(), decoder_dim]),
                         output["pred_acoustic_features"].shape)

    def test_fixed_attention_parallel_decoder(self):
        encoder_dim = 12
        vae_dim = 6
        audio_encoder_dim = 6
        n_frames_per_step = 5
        p_teacher_forcing = 1
        decoder_dim = 15

        model_config = enc_dec_dyn.Config(
            modules=[
                self._get_encoder_config(encoder_dim),
                self._get_encoder_vae_config(decoder_dim, vae_dim),
                self._get_fixed_attention_config(),
                self._get_parallel_decoder_config(encoder_dim + vae_dim,
                                                  decoder_dim)
            ]
        )
        model = model_config.create_model()

        phoneme_seq_length = torch.tensor((10, 12), dtype=torch.long)
        phoneme_max_length = torch.tensor(12, dtype=torch.long)
        seq_length = torch.tensor((100, 75), dtype=torch.long)
        max_length = torch.tensor(100, dtype=torch.long)
        batch_size = 2

        test_input = {}
        test_input["phonemes"] = torch.ones(
            [batch_size, phoneme_seq_length.max(), 1]).long()
        test_input["acoustic_features"] = torch.ones(
            [batch_size, seq_length.max(), decoder_dim])
        test_input["attention_matrix"] = torch.zeros(
            (batch_size, max_length, phoneme_max_length))
        seq_length_dict = {"phonemes": phoneme_seq_length,
                           "acoustic_features": seq_length,
                           "attention_matrix": seq_length}
        max_length_dict = {"phonemes": phoneme_max_length,
                           "acoustic_features": max_length,
                           "attention_matrix": max_length}

        model.init_hidden(batch_size)
        output = model(test_input, seq_length_dict, max_length_dict)

        self.assertEqual(
            torch.Size([batch_size, phoneme_seq_length.max(), encoder_dim]),
            output["phoneme_embeddings"].shape)
        self.assertEqual(
            torch.Size([batch_size, seq_length.max(), vae_dim]),
            output["emb_mu"].shape)
        self.assertEqual(
            torch.Size([batch_size, seq_length.max(), decoder_dim]),
            output["pred_acoustic_features"].shape)
        self.assertTrue(
            (phoneme_seq_length == seq_length_dict["phoneme_embeddings"]).any())
        self.assertTrue(
            (phoneme_max_length == max_length_dict["phoneme_embeddings"]).any())
        self.assertTrue((seq_length == max_length_dict["emb_mu"]).any())
        self.assertTrue(
            (seq_length == seq_length_dict["pred_acoustic_features"]).any())

        expected_params = 1 + 2 * 3  # Phoneme encoder: 1 Emb + 3 Conv (weight & bias)
        expected_params += 2 * 3 + 4 + 1  # Acoustic encoder: 3 Conv + GRU + VAE projection
        expected_params += 2 + 4  # Parallel decoder: Linear + LSTM
        self.assertEqual(expected_params, len([*model.named_parameters()]))
        output["pred_acoustic_features"].sum().backward()

    def test_save_load(self):
        encoder_dim = 12
        vae_dim = 6
        decoder_dim = 15

        def ordered(obj):
            if isinstance(obj, dict):
                return sorted((k, ordered(v)) for k, v in obj.items())
            if isinstance(obj, list):
                return sorted(ordered(x) for x in obj)
            else:
                return obj

        model_config = enc_dec_dyn.Config(
            modules=[
                self._get_encoder_config(encoder_dim),
                self._get_encoder_vae_config(decoder_dim, vae_dim),
                self._get_fixed_attention_config(),
                self._get_parallel_decoder_config(encoder_dim + vae_dim,
                                                  decoder_dim)
            ]
        )
        model = model_config.create_model()

        other_model = model_config.create_model()
        self.assertTrue((list(model.parameters())[15]
                         != list(other_model.parameters())[15]).any())

        config_json = model.get_config_as_json()
        # out_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), type(self).__name__, "test_save_load")
        # makedirs_safe(out_dir)
        # with open(os.path.join(out_dir, "model_config.json"), "w") as f:
        #     f.write(jsonpickle.encode(model_config, indent=4))
        # with open(os.path.join(out_dir, "config.json"), "w") as f:
        #     f.write(config_json)
        self.assertEqual(ordered(jsonpickle.encode(model_config, indent=4)),
                         ordered(config_json))
        params = model.state_dict()
        recreated_config = jsonpickle.decode(config_json)
        recreated_model = recreated_config.create_model()
        recreated_model.load_state_dict(params)

        self.assertTrue((list(model.parameters())[0]
                         == list(recreated_model.parameters())[0]).all())
        # shutil.rmtree(out_dir)

    def test_fixed_attention_auto_regressive_inference(self):
        encoder_dim = 12
        vae_dim = 6
        audio_encoder_dim = 6
        n_frames_per_step = 5
        p_teacher_forcing = 0.5
        decoder_dim = 15

        model_config = enc_dec_dyn.Config(
            modules=[
                self._get_encoder_config(encoder_dim),
                self._get_encoder_embedding_config(vae_dim),
                self._get_fixed_attention_decoder_config(
                    audio_encoder_dim, encoder_dim + audio_encoder_dim + vae_dim,
                    decoder_dim, n_frames_per_step, p_teacher_forcing),
                self._get_postnet_config(decoder_dim)
            ]
        )
        model = model_config.create_model()

        phoneme_seq_length = torch.tensor((10, 12), dtype=torch.long)
        phoneme_max_length = torch.tensor(12, dtype=torch.long)
        seq_length = torch.tensor((100, 75), dtype=torch.long)
        max_length = torch.tensor(100, dtype=torch.long)
        batch_size = 2

        test_input = {}
        test_input["emb_idx"] = torch.zeros([batch_size, 1]).long()
        test_input["acoustic_features"] = torch.ones(
            [batch_size, max_length + 25, decoder_dim])
        test_input["phonemes"] = torch.ones(
            [batch_size, phoneme_seq_length.max(), 1]).long()
        test_input["attention_matrix"] = torch.zeros(
            (batch_size, max_length, phoneme_max_length))
        seq_length_dict = {"phonemes": phoneme_seq_length,
                           "emb_idx": 1,
                           "acoustic_features": max_length + 25}
        max_length_dict = {"phonemes": phoneme_max_length,
                           "emb_idx": 1,
                           "acoustic_features": max_length + 25}
        org_test_input = copy.deepcopy(test_input)

        model.init_hidden(batch_size)
        output = model.inference(test_input, seq_length_dict, max_length_dict)

        output_filtered = {k: v for k, v in output.items()
                           if k not in org_test_input}

        for key in output_filtered.keys():
            self.assertIn(key, seq_length_dict,
                          msg="{} not found in seq_length_dict".format(key))
            self.assertIn(key, max_length_dict,
                          msg="{} not found in max_length_dict".format(key))
        self.assertIn("acoustic_features", test_input,
                      "Teacher forcing key was removed from given dict.")
        self.assertEqual(torch.Size([batch_size, seq_length.max(), decoder_dim]),
                         output["pred_intermediate_acoustic_features"].shape)

        test_input.pop("acoustic_features")
        seq_length_dict.pop("acoustic_features")
        max_length_dict.pop("acoustic_features")

        model.init_hidden(batch_size)
        output = model.inference(test_input, seq_length_dict, max_length_dict)

        self.assertEqual(torch.Size([batch_size, seq_length.max(), decoder_dim]),
                         output["pred_intermediate_acoustic_features"].shape)

    def test_fixed_attention_parallel_decoder_inference(self):
        encoder_dim = 12
        vae_dim = 6
        audio_encoder_dim = 6
        n_frames_per_step = 5
        p_teacher_forcing = 0.5
        decoder_dim = 15

        model_config = enc_dec_dyn.Config(
            modules=[
                self._get_encoder_config(encoder_dim),
                self._get_encoder_embedding_config(vae_dim),
                self._get_fixed_attention_config(),
                self._get_parallel_decoder_config(encoder_dim + vae_dim,
                                                  decoder_dim),
            ]
        )
        model = model_config.create_model()

        phoneme_seq_length = torch.tensor((10, 12), dtype=torch.long)
        phoneme_max_length = torch.tensor(12, dtype=torch.long)
        seq_length = torch.tensor((100, 75), dtype=torch.long)
        max_length = torch.tensor(100, dtype=torch.long)
        batch_size = 2

        test_input = {}
        test_input["emb_idx"] = torch.zeros([batch_size, 1]).long()
        test_input["phonemes"] = torch.ones(
            [batch_size, phoneme_seq_length.max(), 1]).long()
        test_input["attention_matrix"] = torch.zeros(
            (batch_size, max_length, phoneme_max_length))
        seq_length_dict = {"phonemes": phoneme_seq_length, "emb_idx": 1}
        seq_length_dict["attention_matrix"] = seq_length
        max_length_dict = {"phonemes": phoneme_max_length, "emb_idx": 1}
        max_length_dict["attention_matrix"] = max_length
        org_test_input = copy.deepcopy(test_input)

        model.init_hidden(batch_size)
        output = model.inference(test_input, seq_length_dict, max_length_dict)

        output_filtered = {k: v for k, v in output.items() if k not in org_test_input}

        for key in output_filtered.keys():
            self.assertIn(key, seq_length_dict,
                          msg="{} not found in seq_length_dict".format(key))
            self.assertIn(key, max_length_dict,
                          msg="{} not found in max_length_dict".format(key))
        self.assertEqual(torch.Size([batch_size, seq_length.max(), decoder_dim]),
                         output["pred_acoustic_features"].shape)
