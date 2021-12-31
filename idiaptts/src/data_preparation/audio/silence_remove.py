#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#

"""Module description.
Ensures for each given audio file that at max <self.min_silence_ms> ms
are in the front and back of the audio. This value can be adapted by the
--min_silence_ms parameter. Audio files are specified through the
file_id_list parameter. No silence is added.
"""

# System imports.
import argparse
import logging
import os

# Third-party imports.
import librosa
import soundfile

# Local source tree imports.
from idiaptts.misc.utils import makedirs_safe
from idiaptts.src.data_preparation.audio.AudioProcessing import AudioProcessing


class SilenceRemover(object):
    logger = logging.getLogger(__name__)

    def __init__(self, min_silence_ms=200):
        self.min_silence_ms = min_silence_ms

    def process_list(self, id_list, dir_audio, dir_out, format="wav",
                     silence_threshold_db=-50, chunk_size_ms=10):

        for file_id in id_list:
            self.process_file(file_id + "." + format, dir_audio, dir_out,
                              silence_threshold_db, chunk_size_ms)

    @staticmethod
    def _detect_leading_silence(sound, silence_threshold_db=-50.0,
                                chunk_size_ms=10):
        """
        Sound has to be a pydub.AudioSegment. Iterate over chunks until
        the first one with sound is found.

        :param sound:                 A pydub.AudioSegment.
        :param silence_threshold_db:  A maximum db for silence. Use
                                      higher values for noisy
                                      environments.
        :param chunk_size_ms:         Returns the number of leading
                                      silent ms.
        :return:                      Number of leading silent ms.
        """
        trim_ms = 0

        assert chunk_size_ms > 0, "Invalid chunk size."
        while sound[trim_ms:trim_ms+chunk_size_ms].dBFS < silence_threshold_db\
                and trim_ms < len(sound):
            trim_ms += chunk_size_ms

        return trim_ms

    def process_file(self, file, dir_audio, dir_out, silence_threshold_db=-50,
                     hop_size_ms=None):

        raw, fs = soundfile.read(os.path.join(dir_audio, file))

        frame_length = AudioProcessing.fs_to_frame_length(fs)
        if hop_size_ms is None:
            hop_size_ms = min(self.min_silence_ms, 32)

        _, indices = librosa.effects.trim(
            raw,
            top_db=abs(silence_threshold_db),
            frame_length=frame_length,
            hop_length=int(fs / 1000 * hop_size_ms))

        trim_start = indices[0] / fs * 1000
        trim_end = (len(raw) - indices[1]) / fs * 1000

        # Add silence to the front if audio starts to early.
        if trim_start < self.min_silence_ms:
            # TODO: Find a robust way to create silence so that alignment still
            #       works (maybe concat mirrored segments).
            logging.warning(
                "File {} has only {} ms of silence in the beginning.".format(
                    file, trim_start))
            trim_start = 0
        else:
            trim_start -= self.min_silence_ms

        # Append silence if audio ends too late.
        if trim_end < self.min_silence_ms:
            # See TODO above.
            logging.warning(
                "File {} has only {} ms of silence in the end.".format(
                    file, trim_end))
            trim_end = 0
        else:
            trim_end -= self.min_silence_ms

        start_frame = int(trim_start * fs / 1000)
        end_frame = int(-trim_end * fs / 1000 - 1)
        trimmed_raw = raw[start_frame:end_frame]

        out_file = os.path.join(dir_out, file)
        makedirs_safe(os.path.dirname(out_file))
        soundfile.write(out_file, trimmed_raw, samplerate=fs)

        return trimmed_raw


def main():
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-w", "--dir_wav",
                        help="Directory containing the wav files.", type=str,
                        dest="dir_wav", required=True)
    parser.add_argument("-o", "--dir_out",
                        help="Directory to save the trimmed files.", type=str,
                        dest="dir_out", required=True)
    parser.add_argument("-f", "--file_id_list",
                        help="Full path to file containing the ids.", type=str,
                        dest="file_id_list", required=True)
    parser.add_argument("--format", help="Format of the audio file, e.g. WAV.",
                        type=str, dest="format", required=False, default='wav')
    parser.add_argument(
        "--silence_db",
        help="Threshold until which a frame is considered to be silent.",
        type=int, dest="silence_threshold_db", required=False, default=-50)
    parser.add_argument(
        "--chunk_size",
        help="Size of the chunk (frame size) in ms on which db is computed.",
        type=int, dest="chunk_size_ms", required=False, default=10)
    parser.add_argument(
        "--min_silence_ms",
        help="Milliseconds of silence which are always kept in front and back "
        "of audio file.", type=int, dest="min_silence_ms", required=False,
        default=200)

    args = parser.parse_args()

    with open(args.file_id_list) as f:
        id_list = f.readlines()
    id_list[:] = [s.strip(' \t\n\r') for s in id_list]

    makedirs_safe(args.dir_out)

    silence_remover = SilenceRemover(args.min_silence_ms)
    silence_remover.process_list(id_list, args.dir_wav, args.dir_out,
                                 args.format, args.silence_threshold_db,
                                 args.chunk_size_ms)


if __name__ == "__main__":
    main()
