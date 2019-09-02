#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#

"""Module description.
Ensures for each given audio file that at max <self.min_silence_ms> ms are in the front and back of the audio.
This value can be adapted by the --min_silence_ms parameter.
Audio files are specified through the file_id_list parameter.
"""

# System imports.
import os
import logging
import argparse

# Third-party imports.
from pydub import AudioSegment

# Local source tree imports.
from idiaptts.misc.utils import makedirs_safe


class SilenceRemover(object):
    """Class description.
    """
    logger = logging.getLogger(__name__)

    def __init__(self, min_silence_ms=200):
        self.min_silence_ms = min_silence_ms

    def process_list(self, id_list, dir_audio, dir_out, format="wav", silence_threshold_db=-50, chunk_size_ms=10):

        # logging.getLogger("pydub.converter").setLevel(level=logging.INFO)

        for file_id in id_list:
            self.process_file(file_id + "." + format, dir_audio, dir_out, format, silence_threshold_db, chunk_size_ms)

    @staticmethod
    def _detect_leading_silence(sound, silence_threshold_db=-50.0, chunk_size_ms=10):
        """
        Sound has to be a pydub.AudioSegment.
        Silence_threshold in dB, chunk_size in ms.

        Iterate over chunks until you find the first one with sound.

        :param sound:                    A pydub.AudioSegment.
        :param silence_threshold_db:     A maximum db for silence. Use higher values for noisy environments.
        :param chunk_size_ms:            Returns the number of leading silent ms.
        :return:
        """
        trim_ms = 0

        assert chunk_size_ms > 0  # to avoid infinite loop
        while sound[trim_ms:trim_ms+chunk_size_ms].dBFS < silence_threshold_db and trim_ms < len(sound):
            trim_ms += chunk_size_ms

        return trim_ms

    def process_file(self, file, dir_audio, dir_out, audio_format="wav", silence_threshold_db=-50, chunk_size_ms=10):
        sound = AudioSegment.from_file(os.path.join(dir_audio, file), format=audio_format)

        trim_start = self._detect_leading_silence(sound, silence_threshold_db, chunk_size_ms)
        trim_end = self._detect_leading_silence(sound.reverse(), silence_threshold_db, chunk_size_ms)

        # Add silence to the front if audio starts to early.
        if trim_start < self.min_silence_ms:
            # TODO: Find a robust way to create silence so that HTK alignment still works (maybe concat mirrored segments).
            logging.warning("File {} has only {} ms of silence in the beginning.".format(file, trim_start))
            # AudioSegment.silent(duration=self.min_silence_ms-trim_start)
            # if trim_start > 0:
            #     silence = (sound[:trim_start] * (math.ceil(self.min_silence_ms / trim_start) - 1))[:self.min_silence_ms-trim_start]
            #     sound = silence + sound
            # elif trim_end > 0:
            #     silence = (sound[-trim_end:] * (math.ceil(self.min_silence_ms / trim_end) - 1))[:self.min_silence_ms-trim_end]
            #     sound = silence + sound
            # else:
            #     self.logger.warning("Cannot append silence to the front of " + file + ". No silence exists at front or end which can be copied.")
            trim_start = 0
        else:
            trim_start -= self.min_silence_ms

        # Append silence if audio ends too late.
        if trim_end < self.min_silence_ms:
            logging.warning("File {} has only {} ms of silence in the end.".format(file, trim_end))
            # silence = AudioSegment.silent(duration=self.min_silence_ms-trim_end)
            # if trim_end > 0:
            #     silence = (sound[-trim_end:] * (math.ceil(self.min_silence_ms / trim_end) - 1))[:self.min_silence_ms-trim_end]
            #     sound = sound + silence
            # elif trim_start > 0:
            #     silence = (sound[:trim_start] * (math.ceil(self.min_silence_ms / trim_start) - 1))[:self.min_silence_ms-trim_start]
            #     sound = sound + silence
            # else:
            #     self.logger.warning("Cannot append silence to the end of " + file + ". No silence exists at front or end which can be copied.")
            trim_end = 0
        else:
            trim_end -= self.min_silence_ms

        # Trim the sound.
        trimmed_sound = sound[trim_start:-trim_end-1]

        # Save trimmed sound to file.
        out_file = os.path.join(dir_out, file)
        makedirs_safe(os.path.dirname(out_file))
        trimmed_sound.export(out_file, format=audio_format)

        return trimmed_sound


def main():
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-w", "--dir_wav", help="Directory containing the wav files.", type=str,
                        dest="dir_wav", required=True)
    parser.add_argument("-o", "--dir_out", help="Directory to save the trimmed files.", type=str,
                        dest="dir_out", required=True)
    parser.add_argument("-f", "--file_id_list", help="Full path to file containing the ids.", type=str,
                        dest="file_id_list", required=True)
    parser.add_argument("--format", help="Format of the audio file, e.g. WAV.", type=str,
                        dest="format", required=False, default='wav')
    parser.add_argument("--silence_db", help="Threshold until which a frame is considered to be silent.", type=int,
                        dest="silence_threshold_db", required=False, default=-50)
    parser.add_argument("--chunk_size", help="Size of the chunk (frame size) in ms on which db is computed.", type=int,
                        dest="chunk_size_ms", required=False, default=10)
    parser.add_argument("--min_silence_ms", help="Milliseconds of silence which are always kept in front and back of audio file.", type=int,
                        dest="min_silence_ms", required=False, default=200)

    # Parse arguments
    args = parser.parse_args()

    # Read which files to process.
    with open(args.file_id_list) as f:
        id_list = f.readlines()
    # Trim entries in-place.
    id_list[:] = [s.strip(' \t\n\r') for s in id_list]

    # Create output directory if missing.
    makedirs_safe(args.dir_out)

    # Start silence removal.
    silence_remover = SilenceRemover(args.min_silence_ms)
    silence_remover.process_list(id_list, args.dir_wav, args.dir_out,
                                 args.format, args.silence_threshold_db, args.chunk_size_ms)


if __name__ == "__main__":
    main()
