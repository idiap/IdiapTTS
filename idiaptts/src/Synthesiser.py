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
    def run_world_synth(synth_output, hparams):
        """Run the WORLD synthesize method."""

        fft_size = pyworld.get_cheaptrick_fft_size(hparams.synth_fs)

        save_dir = hparams.synth_dir if hparams.synth_dir is not None\
                                     else hparams.out_dir if hparams.out_dir is not None\
                                     else os.path.curdir
        for id_name, output in synth_output.items():
            logging.info("Synthesise {} with the WORLD vocoder.".format(id_name))

            coded_sp, lf0, vuv, bap = WorldFeatLabelGen.convert_to_world_features(output,
                                                                                  contains_deltas=False,
                                                                                  num_coded_sps=hparams.num_coded_sps)
            sp = WorldFeatLabelGen.decode_sp(coded_sp, hparams.sp_type, hparams.synth_fs,
                                             post_filtering=hparams.do_post_filtering).astype(np.double, copy=False)

            f0 = np.exp(lf0, dtype=np.float64)
            vuv[f0 < WorldFeatLabelGen.f0_silence_threshold] = 0  # WORLD throws an error for too small f0 values.
            f0[vuv == 0] = 0.0
            ap = pyworld.decode_aperiodicity(np.ascontiguousarray(bap.reshape(-1, 1), np.float64),
                                             hparams.synth_fs,
                                             fft_size)

            waveform = pyworld.synthesize(f0, sp, ap, hparams.synth_fs)
            waveform = waveform.astype(np.float32, copy=False)  # Does inplace conversion, if possible.

            # Always save as wav file first and convert afterwards if necessary.
            file_path = os.path.join(save_dir, "{}{}{}{}".format(os.path.basename(id_name),
                                                                 "_" + hparams.model_name if hparams.model_name is not None else "",
                                                                 hparams.synth_file_suffix, "_WORLD"))
            makedirs_safe(hparams.synth_dir)
            soundfile.write(file_path + ".wav", waveform, hparams.synth_fs)

            # Use PyDub for special audio formats.
            if hparams.synth_ext.lower() != 'wav':
                as_wave = pydub.AudioSegment.from_wav(file_path + ".wav")
                file = as_wave.export(file_path + "." + hparams.synth_ext, format=hparams.synth_ext)
                file.close()
                os.remove(file_path + ".wav")

    @staticmethod
    def synth_ref(hparams, file_id_list, feature_dir=None):
        # Create reference audio files containing only the vocoder degradation.
        logging.info("Synthesise references with {} for [{}]."
                     .format(hparams.synth_vocoder, ", ".join([id_name for id_name in file_id_list])))

        synth_dict = dict()
        old_synth_file_suffix = hparams.synth_file_suffix
        hparams.synth_file_suffix = '_ref'
        if hparams.synth_vocoder == "WORLD":
            for id_name in file_id_list:
                # Load reference audio features.
                try:
                    output = WorldFeatLabelGen.load_sample(id_name, feature_dir, num_coded_sps=hparams.num_coded_sps)
                except FileNotFoundError as e1:
                    try:
                        output = WorldFeatLabelGen.load_sample(id_name, feature_dir, add_deltas=True,
                                                               num_coded_sps=hparams.num_coded_sps)
                        coded_sp, lf0, vuv, bap = WorldFeatLabelGen.convert_to_world_features(
                                                       output,
                                                       contains_deltas=True,
                                                       num_coded_sps=hparams.num_coded_sps)
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
            hparams.synth_file_suffix += str(hparams.num_coded_sps) + 'sp'
            Synthesiser.run_world_synth(synth_dict, hparams)
        elif hparams.synth_vocoder == "raw":
            for id_name in file_id_list:
                # Use extracted data. Useful to create a reference.
                raw = RawWaveformLabelGen.load_sample(id_name, hparams.frame_rate_output_Hz)
                synth_dict[id_name] = raw
            Synthesiser.run_raw_synth(synth_dict, hparams)
        else:
            raise NotImplementedError("Unknown vocoder type {}.".format(hparams.synth_vocoder))

        # Restore identifier.
        hparams.synth_file_suffix = old_synth_file_suffix

    @staticmethod
    def run_raw_synth(synth_output, hparams):
        """Use Pydub to synthesis audio from raw data given in the synth_output dictionary."""

        for id_name, raw in synth_output.items():
            logging.info("Save {} from raw waveform.".format(id_name))

            # Load raw data into pydub AudioSegment.
            # raw /= raw.abs().max()
            raw *= math.pow(2, hparams.bit_depth) / 2  # Expand to pydub range.
            raw = raw.astype(np.int16)
            audio_seg = AudioSegment(
                # raw audio data (bytes)
                data=raw.tobytes(),
                # 2 byte (16 bit) samples
                sample_width=2,
                # Hz frame rate
                frame_rate=hparams.frame_rate_output_Hz,
                # mono
                channels=1
            )
            audio_seg.set_frame_rate(hparams.synth_fs)

            # Save the audio.
            wav_file_path = os.path.join(hparams.synth_dir, "".join((os.path.basename(id_name).rsplit('.', 1)[0], "_",
                                                                     hparams.model_name, hparams.synth_file_suffix, ".",
                                                                     hparams.synth_ext)))
            audio_seg.export(wav_file_path, format=hparams.synth_ext)

    @staticmethod
    def run_r9y9wavenet_mulaw_world_feats_synth(synth_output, hparams):
        # Import ModelHandlerPyTorch here to prevent circular dependencies.
        from idiaptts.src.neural_networks.pytorch.ModelHandlerPyTorch import ModelHandlerPyTorch

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
            org_frame_rate_output_Hz = hparams.frame_rate_output_Hz
            hparams.frame_rate_output_Hz = 16000
        else:
            org_frame_rate_output_Hz = None
            hparams.add_hparam("frame_rate_output_Hz", 16000)

        org_model_type = hparams.model_type
        hparams.model_type = WaveNetWrapper.IDENTIFIER

        synth_output = copy.copy(synth_output)

        input_fs_Hz = 1000.0 / hparams.frame_size_ms
        in_to_out_multiplier = hparams.frame_rate_output_Hz / input_fs_Hz
        # dir_world_features = os.path.join(self.OutputGen.dir_labels, self.dir_extracted_acoustic_features)
        input_gen = WorldFeatLabelGen(None,
                                      add_deltas=False,
                                      sampling_fn=partial(sample_linearly,
                                                          in_to_out_multiplier=in_to_out_multiplier,
                                                          dtype=np.float32))
        # Load normalisation parameters for wavenet input.
        try:
            norm_params_path = os.path.splitext(hparams.synth_vocoder_path)[0] + "_norm_params.npy"
            input_gen.norm_params = np.load(norm_params_path).reshape(2, -1)
        except FileNotFoundError:
            logging.error("Cannot find normalisation parameters for WaveNet input at {}."
                          "Please save them there with numpy.save().".format(norm_params_path))
            raise

        wavenet_model_handler = ModelHandlerPyTorch()
        wavenet_model_handler.model, *_ = wavenet_model_handler.load_model(hparams.synth_vocoder_path,
                                                                           hparams,
                                                                           verbose=False)

        for id_name, output in synth_output.items():
            logging.info("Synthesise {} with {} vocoder.".format(id_name, hparams.synth_vocoder_path))

            if hparams.do_post_filtering:
                coded_sp, lf0, vuv, bap = input_gen.convert_to_world_features(output,
                                                                              contains_deltas=input_gen.add_deltas,
                                                                              num_coded_sps=hparams.num_coded_sps)
                coded_sp = merlin_post_filter(coded_sp, WorldFeatLabelGen.fs_to_mgc_alpha(hparams.synth_fs))
                output = input_gen.convert_from_world_features(coded_sp, lf0, vuv, bap)

            output = input_gen.preprocess_sample(output)

            # output (T x C) --transpose--> (C x T) --unsqueeze(0)--> (B x C x T).
            output = output.transpose()[None, ...]
            # Wavenet input has to be (B x C x T).
            output, _ = wavenet_model_handler.forward(output, hparams, batch_seq_lengths=(output.shape[-1],))
            output = output[0].transpose()  # Remove batch dim and transpose back to (T x C).
            # Revert mu-law quantization.
            output = output.argmax(axis=1)
            synth_output[id_name] = RawWaveformLabelGen.mu_law_companding_reversed(output, hparams.mu)

        if hasattr(hparams, 'bit_depth'):
            org_bit_depth = hparams.bit_depth
            hparams.bit_depth = 16
        else:
            org_bit_depth = None
            hparams.add_hparam("bit_depth", 16)

        # Add identifier to suffix.
        old_synth_file_suffix = hparams.synth_file_suffix
        hparams.synth_file_suffix += '_' + hparams.synth_vocoder

        Synthesiser.run_raw_synth(synth_output, hparams)

        # Restore identifier.
        hparams.model_type = org_model_type
        hparams.setattr_no_type_check("synth_file_suffix", old_synth_file_suffix)  # Can be None, thus no type check.
        hparams.setattr_no_type_check("bit_depth", org_bit_depth)  # Can be None, thus no type check.
        # TODO: Convert to requested frame rate. if org_frame_rate_output_Hz != 16000:
        hparams.setattr_no_type_check("frame_rate_output_Hz", org_frame_rate_output_Hz)  # Can be None.
