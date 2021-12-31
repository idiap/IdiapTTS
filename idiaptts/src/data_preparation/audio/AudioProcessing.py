#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#

"""Module description:
   Provides various methods to extract audio features like spectrum, cepstrum,
   and variants of them and method for their backwards conversion.
"""

# System imports.
import logging

# Third-party imports.
import librosa
import librosa.display
from nnmnkwii.postfilters import merlin_post_filter
import numpy as np
import pysptk
import pyworld
import scipy
import soundfile

# Local source tree imports.


class AudioProcessing:
    mgc_gamma = -1./3.

    @staticmethod
    def fs_to_mgc_alpha(fs):
        """
        Convert sampling rate to appropriate MGC warping parameter.

        Code base on:
        Merlin's /misc/scripts/vocoder/world/extract_features_for_merlin.sh
        """
        return pysptk.util.mcepalpha(fs)
        # if fs == 16000:
        #     return 0.58
        # elif fs == 22050:
        #     return 0.65
        # elif fs == 44100:
        #     return 0.76
        # elif fs == 48000:
        #     return 0.77
        # else:
        #     raise NotImplementedError()

    @staticmethod
    def fs_to_frame_length(fs):
        """
        Convert sampling rate to frame length for STFT frame length.

        Code base on:
        Merlin's /misc/scripts/vocoder/world/extract_features_for_merlin.sh
        """
        return pyworld.get_cheaptrick_fft_size(fs)  # Better alternative.
        #
        # if fs == 16000 or fs == 22050 or fs == 24000:
        #     return 1024
        # elif fs == 44100 or fs == 48000:
        #     return 2048
        # else:
        #     raise NotImplementedError()

    @staticmethod
    def fs_to_num_bap(fs: int):
        return pyworld.get_num_aperiodicities(fs)

    @staticmethod
    def framing(signal, sample_rate, frame_size_ms=None, frame_hop_ms=5):

        if frame_size_ms is None:
            frame_length = AudioProcessing.fs_to_frame_length(sample_rate)
        else:
            frame_size = frame_size_ms / 1000.
            # Convert from seconds to samples.
            frame_length = frame_size * sample_rate

        frame_stride = frame_hop_ms / 1000.
        frame_step = frame_stride * sample_rate

        # Framing.
        signal_length = len(signal)
        frame_length = int(round(frame_length))
        frame_step = int(round(frame_step))
        num_frames = float(np.abs(signal_length - frame_length)) / frame_step
        # Make sure that we have at least 1 frame.
        num_frames = int(np.ceil(num_frames))

        pad_signal_length = num_frames * frame_step + frame_length
        z = np.zeros((pad_signal_length - signal_length))
        # Pad Signal to make sure that all frames have equal number of
        # samples without truncating any samples from the original signal.
        pad_signal = np.append(signal, z)

        indices = np.tile(np.arange(0, frame_length), (num_frames, 1))\
            + np.tile(np.arange( 0, num_frames * frame_step, frame_step),
                      (frame_length, 1)).T
        frames = pad_signal[indices.astype(np.int32, copy=False)]

        return frames

    @staticmethod
    def get_raw(audio_name: str, preemphasis: float = 0.0):
        """
        Extract the raw audio in [-1, 1] and apply pre-emphasis. 0.0
        pre-emphasis means no pre-emphasis.
        """
        # librosa_raw = librosa.load(audio_name, sr=16000)  # raw in [-1, 1]
        # fs, raw = wavfile.read(audio_name)  # raw in [-32768, 32768]
        raw, fs = soundfile.read(audio_name)  # raw in [-1, 1]

        # Pre-emphasis
        raw = np.append(raw[0], raw[1:] - preemphasis * raw[:-1])

        return raw, fs

    @staticmethod
    def extract_mgc(amp_sp: np.array, fs: int = None, num_coded_sps: int = 60,
                    mgc_alpha: float = None) -> np.array:
        """Extract MGC from the amplitude spectrum from SPTK."""

        if mgc_alpha is None:
            assert fs is not None, "Either sampling rate or mgc alpha has to be given."
            mgc_alpha = AudioProcessing.fs_to_mgc_alpha(fs)

        mgc = pysptk.mgcep(amp_sp,
                           order=num_coded_sps - 1,
                           alpha=mgc_alpha,
                           gamma=AudioProcessing.mgc_gamma,
                           eps=1.0e-8,
                           min_det=0.0,
                           etype=1,
                           itype=3)

        return mgc.astype(np.float32, copy=False)

    @staticmethod
    def extract_mcep(amp_sp: np.array, num_coded_sps: int, mgc_alpha: float) \
            -> np.array:
        """Extract MCep from the amplitude spectrum with SPTK."""
        mcep = pysptk.mcep(amp_sp,
                           order=num_coded_sps - 1,
                           alpha=mgc_alpha,
                           eps=1.0e-8,
                           min_det=0.0,
                           etype=1,
                           itype=3)
        return mcep.astype(np.float32, copy=False)

    @staticmethod
    def librosa_extract_amp_sp(raw: np.array,
                               fs: int, n_fft: int = None,
                               hop_size_ms: int = 5,
                               win_length_ms: int = None,
                               window: str = "hann",
                               center: bool = True,
                               pad_mode: str = 'reflect') -> np.array:
        """
        Extract amplitude spectrum from raw [-1, 1] signal. Parameters
        are explained in librosa.stft.
        """

        if n_fft is None:
            assert fs is not None, "Either fs or n_fft has to be given."
            n_fft = AudioProcessing.fs_to_frame_length(fs)

        if win_length_ms is None:
            win_length = None
        else:
            win_length = int(win_length_ms / 1000. * fs)

        amp_sp = np.abs(librosa.stft(raw,
                                     n_fft=n_fft,
                                     hop_length=int(hop_size_ms / 1000. * fs),
                                     win_length=win_length,
                                     center=center, window=window,
                                     pad_mode=pad_mode))

        return (amp_sp / np.sqrt(amp_sp.shape[0])).T  # T x n_fft

    @staticmethod
    def extract_mfbanks(raw: np.array = None, fs: int = 22050,
                        amp_sp: np.array = None, n_fft: int = None,
                        hop_size_ms: int = 5, num_coded_sps: int = 80,
                        win_length_ms: int = None) -> np.array:
        """
        Extract Mel-filter banks using librosa.

        :param raw:            Raw audio signal in [-1, 1], ignored when
                               amp_sp is given.
        :param fs:             Sampling rate
        :param amp_sp:         Amplitude spectrum, if not given it is
                               extracted with librosa from the raw input.
        :param n_fft:          FFT length
        :param hop_size_ms:    Hop size in milliseconds.
        :param num_coded_sps:  Number of output mel-filter banks
        :return:               Mel-filter banks as float32.
        """
        assert (n_fft is not None or amp_sp is not None), \
            "Either FFT size has to be given or amplitude spectrogram."
        if amp_sp is None:
            assert raw is not None, "Either raw signal or amplitude "\
                "spectrum must be given."
            amp_sp = AudioProcessing.librosa_extract_amp_sp(
                raw, fs, n_fft, hop_size_ms, win_length_ms)
        if num_coded_sps == -1:
            mfbanks = amp_sp
        else:
            if win_length_ms is None:
                win_length = None
            else:
                win_length = int(win_length_ms / 1000. * fs)

            # TODO: Include fmin=50?
            mfbanks = librosa.feature.melspectrogram(
                sr=fs,
                S=amp_sp.T,  # Use amplitude spectrum.
                n_fft=n_fft,
                hop_length=int(fs * hop_size_ms / 1000.0),
                win_length=win_length,
                n_mels=num_coded_sps).T

        return mfbanks.astype(np.float32, copy=False)

    # @staticmethod
    # def extract_mfcc(raw: np.array, fs: int, amp_sp: np.array = None,
    #                  n_fft: int = None, hop_size_ms: int = 5,
    #                  num_coded_sps = 12) -> np.array:
    #     # Using the default number (128) of mel bins.
    #     mel_sp = librosa.feature.melspectrogram(
    #         y=raw,  # Ignored when amp_sp is not None.
    #         sr=fs,
    #         S=amp_sp.T if amp_sp is not None else None,  # Use amplitude spectrum.
    #         n_fft=n_fft,
    #         hop_length=int(fs * hop_size_ms / 1000.0))
    #     log_pow_mel_sp = librosa.power_to_db(np.square(mel_sp), top_db=None)
    #     mfcc = librosa.feature.mfcc(sr=fs, S=log_pow_mel_sp,
    #                                 n_mfcc=num_coded_sps).T

    #     return mfcc

    @staticmethod
    def mcep_to_amp_sp(mcep: np.array, fs: int, alpha: float = None):
        """Convert MCep back to amplitude spectrum using SPTK."""
        if alpha is None:
            alpha = AudioProcessing.fs_to_mgc_alpha(fs)
        amp_sp = pysptk.mgc2sp(np.ascontiguousarray(mcep, dtype=np.float64),
                               alpha=alpha,
                               gamma=0.0,
                               fftlen=AudioProcessing.fs_to_frame_length(fs))
        return np.exp(amp_sp.real.astype(np.float32, copy=False))

    @staticmethod
    def mgc_to_amp_sp(mgc: np.array, fs: int, alpha: float = None,
                      gamma: float = None, n_fft: int = None):
        """Convert MGCs back to amplitude spectrum using SPTK."""
        if alpha is None:
            alpha = AudioProcessing.fs_to_mgc_alpha(fs)
        if gamma is None:
            gamma = AudioProcessing.mgc_gamma
        if n_fft is None:
            n_fft = AudioProcessing.fs_to_frame_length(fs)
        amp_sp = pysptk.mgc2sp(np.ascontiguousarray(mgc, dtype=np.float64),
                               alpha=alpha,
                               gamma=gamma,
                               fftlen=n_fft)

        # WORLD expects spectrum divided by number of final bins, but
        # SPTK does not divide it. TODO: Is this commend up to date?
        return np.exp(amp_sp.real.astype(np.float32, copy=False))

    @staticmethod
    def amp_sp_to_raw(amp_sp: np.array, fs: int, hop_size_ms: int = 5,
                      preemphasis: float = 0.00):
        """
        Transform the amplitude spectrum into the waveform with
        Griffin-Lim. The amplitude spectrum has to have the pitch
        information. Using amplitude spectrum which was extracted with
        pitch aligned windows (as WORLD does it) will not work.
        """
        raw = librosa.griffinlim(amp_sp.T * np.sqrt(amp_sp.shape[1]),
                                 hop_length=int(fs * hop_size_ms / 1000.))
        return AudioProcessing.depreemphasis(raw, preemphasis)

    @staticmethod
    def mfbanks_to_amp_sp(coded_sp: np.array, fs: int, n_fft: int = None):
        """
        Convert Mel-filter banks back to amplitude spectrum. This does
        not work well. Use an SSRN instead.
        """
        if n_fft is None:
            n_fft = AudioProcessing.fs_to_frame_length(fs)
        amp_sp = librosa.feature.inverse.mel_to_stft(coded_sp.T, sr=fs,
                                                     n_fft=n_fft, power=1.0,
                                                     norm=None).T
        return amp_sp * amp_sp.shape[1]

    @staticmethod
    def decode_sp(coded_sp: np.array, sp_type: str = "mcep", fs: int = None,
                  alpha: float = None, mgc_gamma: float = None,
                  n_fft: int = None, post_filtering: bool = False):

        if post_filtering:
            if sp_type in ["mcep", "mgc"]:
                coded_sp = merlin_post_filter(
                    coded_sp, AudioProcessing.fs_to_mgc_alpha(fs))
            else:
                logging.warning(
                    "Post-filtering only implemented for cepstrum features.")

        if sp_type == "mcep":
            return AudioProcessing.mcep_to_amp_sp(coded_sp, fs, alpha)
        elif sp_type == "mgc":
            return AudioProcessing.mgc_to_amp_sp(coded_sp, fs, alpha,
                                                 mgc_gamma, n_fft)
        elif sp_type == "mfbanks":
            return AudioProcessing.mfbanks_to_amp_sp(coded_sp, fs, n_fft)
        elif sp_type == "amp_sp":
            return coded_sp
        else:
            raise NotImplementedError("Unknown feature type {}. No decoding "
                                      "method available.".format(sp_type))

    @staticmethod
    def depreemphasis(raw: np.ndarray, preemphasis: float):
        return scipy.signal.lfilter([1], [1, -preemphasis], raw)

    @staticmethod
    def amp_to_db(amp_sp):
        return 20 * np.log10(np.maximum(1e-5, amp_sp))

    @staticmethod
    def db_to_amp(log_amp_sp):
        return np.power(10.0, log_amp_sp * 0.05)
