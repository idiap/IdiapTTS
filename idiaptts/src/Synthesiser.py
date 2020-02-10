#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#


# System imports.
import copy
import logging
from functools import partial
import numpy as np
import os
import math

# Third-party imports.
import pydub
import soundfile
from nnmnkwii.postfilters import merlin_post_filter
from pydub import AudioSegment
from pyworld import pyworld

# Local source tree imports.
from idiaptts.src.data_preparation.world.WorldFeatLabelGen import WorldFeatLabelGen
from idiaptts.misc.utils import makedirs_safe, sample_linearly
from idiaptts.src.data_preparation.audio.RawWaveformLabelGen import RawWaveformLabelGen
from idiaptts.src.neural_networks.pytorch.models.WaveNetWrapper import WaveNetWrapper


class Synthesiser(object):

    @staticmethod
    def run_world_synth(synth_output, hparams, use_model_name=True):
        """Run the WORLD synthesize method."""

        fft_size = pyworld.get_cheaptrick_fft_size(hparams.synth_fs)

        save_dir = hparams.synth_dir if hparams.synth_dir is not None\
                                     else hparams.out_dir if hparams.out_dir is not None\
                                     else os.path.curdir
        for id_name, output in synth_output.items():
            logging.info("Synthesise {} with the WORLD vocoder.".format(id_name))

            coded_sp, lf0, vuv, bap = WorldFeatLabelGen.convert_to_world_features(output,
                                                                                  contains_deltas=False,
                                                                                  num_coded_sps=hparams.num_coded_sps,
                                                                                  num_bap=hparams.num_bap)
            amp_sp = WorldFeatLabelGen.decode_sp(coded_sp, hparams.sp_type, hparams.synth_fs,
                                                 post_filtering=hparams.do_post_filtering).astype(np.double, copy=False)
            args = dict()
            for attr in "preemphasize", "f0_silence_threshold", "lf0_zero":
                if hasattr(hparams, attr):
                    args[attr] = getattr(hparams, attr)
            waveform = WorldFeatLabelGen.world_features_to_raw(amp_sp, lf0, vuv, bap,
                                                               fs=hparams.synth_fs, n_fft=fft_size, **args)

            # Always save as wav file first and convert afterwards if necessary.
            file_path = os.path.join(save_dir, "{}{}{}{}"
                                     .format(os.path.basename(id_name),
                                             "_" + hparams.model_name if use_model_name
                                                                         and hparams.model_name is not None
                                                                      else "",
                                             hparams.synth_file_suffix,
                                             "_WORLD"))
            makedirs_safe(hparams.synth_dir)
            soundfile.write(file_path + ".wav", waveform, hparams.synth_fs)

            # Use PyDub for special audio formats.
            if hparams.synth_ext.lower() != 'wav':
                as_wave = pydub.AudioSegment.from_wav(file_path + ".wav")
                file = as_wave.export(file_path + "." + hparams.synth_ext, format=hparams.synth_ext)
                file.close()
                os.remove(file_path + ".wav")

    @staticmethod
    def copy_synth(hparams, file_id_list, feature_dir=None):
        # Create reference audio files containing only the vocoder degradation.
        logging.info("Copy synthesis with {} for [{}]."
                     .format(hparams.synth_vocoder, ", ".join([id_name for id_name in file_id_list])))

        synth_dict = dict()
        old_synth_file_suffix = hparams.synth_file_suffix
        hparams.synth_file_suffix = '_ref'
        if hparams.synth_vocoder == "WORLD":
            for id_name in file_id_list:
                # Load reference audio features.
                try:
                    output = WorldFeatLabelGen.load_sample(id_name,
                                                           feature_dir,
                                                           num_coded_sps=hparams.num_coded_sps,
                                                           num_bap=hparams.num_bap,
                                                           sp_type=hparams.sp_type)
                except FileNotFoundError as e1:
                    try:
                        output = WorldFeatLabelGen.load_sample(id_name,
                                                               feature_dir,
                                                               add_deltas=True,
                                                               num_coded_sps=hparams.num_coded_sps,
                                                               num_bap=hparams.num_bap,
                                                               sp_type=hparams.sp_type)
                        coded_sp, lf0, vuv, bap = WorldFeatLabelGen.convert_to_world_features(
                                                       output,
                                                       contains_deltas=True,
                                                       num_coded_sps=hparams.num_coded_sps,
                                                       num_bap=hparams.num_bap)
                        length = len(output)
                        lf0 = lf0.reshape(length, 1)
                        vuv = vuv.reshape(length, 1)
                        bap = bap.reshape(length, 1)
                        output = np.concatenate((coded_sp, lf0, vuv, bap), axis=1)
                    except FileNotFoundError as e2:
                        logging.error("Cannot find extracted WORLD features with or without deltas in {}."
                                      .format(feature_dir))
                        raise Exception([e1, e2])
                synth_dict[id_name] = output

            # Add identifier to suffix.
            old_synth_file_suffix = hparams.synth_file_suffix
            hparams.synth_file_suffix += str(hparams.num_coded_sps) + hparams.sp_type
            Synthesiser.run_world_synth(synth_dict, hparams, use_model_name=False)
        elif hparams.synth_vocoder == "raw" or hparams.synth_vocoder.starts_with("r9y9wavenet"):
            for id_name in file_id_list:
                # Use extracted data. Useful to create a reference.
                raw = RawWaveformLabelGen.load_sample(id_name, hparams.frame_rate_output_Hz)
                synth_dict[id_name] = raw
            Synthesiser.run_raw_synth(synth_dict, hparams, use_model_name=False)
        else:
            raise NotImplementedError("Unknown vocoder type {}.".format(hparams.synth_vocoder))

        # Restore identifier.
        hparams.synth_file_suffix = old_synth_file_suffix

    @staticmethod
    def run_raw_synth(synth_output, hparams, use_model_name=True):
        """Use Pydub to synthesis audio from raw data given in the synth_output dictionary."""

        for id_name, raw in synth_output.items():
            # Save the audio.
            wav_file_path = os.path.join(hparams.synth_dir, "".join((os.path.basename(id_name).rsplit('.', 1)[0],
                                                                     "_" + hparams.model_name if use_model_name else "",
                                                                     hparams.synth_file_suffix,
                                                                     ".",
                                                                     hparams.synth_ext)))
            Synthesiser.raw_to_file(wav_file_path, raw, hparams.synth_fs, hparams.bit_depth)

    @staticmethod
    def raw_to_file(file_path, raw, fs, bit_depth):
        logging.info("Save {} from raw waveform.".format(file_path))

        # Load raw data into pydub AudioSegment.
        # raw /= raw.abs().max()
        raw *= math.pow(2, bit_depth) / 2  # Expand to pydub range.
        raw = raw.astype(np.int16)
        audio_seg = AudioSegment(
            # raw audio data (bytes)
            data=raw.tobytes(),
            # 2 byte (16 bit) samples
            sample_width=2,
            # Hz frame rate
            frame_rate=fs,
            # mono
            channels=1
        )
        audio_seg.set_frame_rate(fs)
        file = audio_seg.export(file_path, format=os.path.splitext(file_path)[1].lstrip('.'))
        file.close()

    @staticmethod
    def run_r9y9wavenet_mulaw_world_feats_synth(synth_output, hparams):

        hparams = copy.deepcopy(hparams)

        # If no path is given, use pre-trained model.
        if not hasattr(hparams, "synth_vocoder_path") or hparams.synth_vocoder_path is None:
            parent_dirs = os.path.realpath(__file__).split(os.sep)
            dir_root = str.join(os.sep, parent_dirs[:parent_dirs.index("IdiapTTS") + 1])
            hparams.synth_vocoder_path = os.path.join(dir_root, "idiaptts", "misc", "pretrained",
                                                      "r9y9wavenet_quantized_16k_world_feats_English.nn")

        # Default quantization is with mu=255.
        if not hasattr(hparams, "mu") or hparams.mu is None:
            hparams.add_hparam("mu", 255)

        if hasattr(hparams, 'frame_rate_output_Hz'):
            hparams.frame_rate_output_Hz = 16000
        else:
            hparams.add_hparam("frame_rate_output_Hz", 16000)

        synth_output = copy.copy(synth_output)

        if hparams.do_post_filtering:
            for id_name, output in synth_output.items():
                coded_sp, lf0, vuv, bap = WorldFeatLabelGen.convert_to_world_features(
                                                                output,
                                                                contains_deltas=False,
                                                                num_coded_sps=hparams.num_coded_sps,
                                                                num_bap=hparams.num_bap)
                coded_sp = merlin_post_filter(coded_sp, WorldFeatLabelGen.fs_to_mgc_alpha(hparams.synth_fs))
                synth_output[id_name] = WorldFeatLabelGen.convert_from_world_features(coded_sp, lf0, vuv, bap)

        if hasattr(hparams, 'bit_depth'):
            hparams.bit_depth = 16
        else:
            hparams.add_hparam("bit_depth", 16)

        Synthesiser.run_wavenet_vocoder(synth_output, hparams)


    @staticmethod
    def run_wavenet_vocoder(synth_output, hparams):
        # Import ModelHandlerPyTorch here to prevent circular dependencies.
        from idiaptts.src.neural_networks.pytorch.ModelHandlerPyTorch import ModelHandlerPyTorch

        assert hparams.synth_vocoder_path is not None, "Please set path to neural vocoder in hparams.synth_vocoder_path"

        hparams = copy.deepcopy(hparams)
        hparams.del_hparam("ignore_layers")
        hparams.model_type = "r9y9WaveNet"

        # Add identifier to suffix.
        hparams.synth_file_suffix += '_' + hparams.synth_vocoder

        if not hasattr(hparams, 'bit_depth'):
            hparams.add_hparam("bit_depth", 16)

        synth_output = copy.copy(synth_output)

        input_fs_Hz = 1000.0 / hparams.frame_size_ms
        assert hasattr(hparams, "frame_rate_output_Hz") and hparams.frame_rate_output_Hz is not None, \
            "hparams.frame_rate_output_Hz has to be set and match the trained WaveNet."
        in_to_out_multiplier = hparams.frame_rate_output_Hz / input_fs_Hz
        # # dir_world_features = os.path.join(self.OutputGen.dir_labels, self.dir_extracted_acoustic_features)
        input_gen = WorldFeatLabelGen(None,
                                      add_deltas=False,
                                      sampling_fn=partial(sample_linearly,
                                                          in_to_out_multiplier=in_to_out_multiplier,
                                                          dtype=np.float32),
                                      num_coded_sps=hparams.num_coded_sps,
                                      num_bap=hparams.num_bap)
        # Load normalisation parameters for wavenet input.
        try:
            norm_params_path = os.path.splitext(hparams.synth_vocoder_path)[0] + "_norm_params.npy"
            input_gen.norm_params = np.load(norm_params_path).reshape(2, -1)
        except FileNotFoundError:
            logging.error("Cannot find normalisation parameters for WaveNet input at {}."
                          "Please save them there with numpy.save().".format(norm_params_path))
            raise

        model_handler = ModelHandlerPyTorch()
        model_handler.model, *_ = model_handler.load_model(hparams.synth_vocoder_path,
                                                           hparams,
                                                           verbose=False)

        for id_name, output in synth_output.items():
            logging.info("Synthesise {} with {} vocoder.".format(id_name, hparams.synth_vocoder_path))

            # Any other post-processing could be done here.

            # Normalize input.
            output = input_gen.preprocess_sample(output)

            # output (T x C) --transpose--> (C x T) --unsqueeze(0)--> (B x C x T).
            output = output.transpose()[None, ...]
            # Wavenet input has to be (B x C x T).
            output, _ = model_handler.forward(output, hparams, batch_seq_lengths=(output.shape[-1],))
            # output, _ = model_handler.forward(output[:, :, :1000], hparams, batch_seq_lengths=(1000,))  # DEBUG
            output = output[0].transpose()  # Remove batch dim and transpose back to (T x C).

            out_channels = output.shape[1]
            if out_channels > 1:  # Check if the output is one-hot (quantized) or 1 (raw).
                # Revert mu-law quantization.
                output = output.argmax(axis=1)
                synth_output[id_name] = RawWaveformLabelGen.mu_law_companding_reversed(output, out_channels)

            # Save the audio.
            wav_file_path = os.path.join(hparams.synth_dir,
                                         "".join((os.path.basename(id_name).rsplit('.', 1)[0], "_",
                                                  hparams.model_name, hparams.synth_file_suffix, ".",
                                                  hparams.synth_ext)))
            Synthesiser.raw_to_file(wav_file_path, synth_output[id_name], hparams.synth_fs, hparams.bit_depth)

        # TODO: Convert to requested frame rate. if org_frame_rate_output_Hz != 16000. This only holds for 16kHz wavenet.
