#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#

"""Module description.
Loudness normalization of each given audio file to a reference average.
Audio files are specified through the file_id_list parameter.
"""

# System imports.
import os
import logging
import argparse

# Third-party imports.
import math
import soundfile

# Local source tree imports.
from idiaptts.misc.utils import makedirs_safe


class LoudnessNormalizer(object):
    logger = logging.getLogger(__name__)

    def __init__(self, ref_rms=0.1):
        self.ref_rms = ref_rms

    def process_list(self, id_list, dir_audio, dir_out, format="wav"):

        for file_id in id_list:
            self.process_file(file_id + "." + format, dir_audio, dir_out)

    def process_file(self, file, dir_audio, dir_out):

        raw, fs = soundfile.read(os.path.join(dir_audio, file))

        raw -= raw.mean()
        raw *= math.sqrt(len(raw) * self.ref_rms**2 / (raw**2).sum())

        out_file = os.path.join(dir_out, file)
        makedirs_safe(os.path.dirname(out_file))
        soundfile.write(out_file, raw, samplerate=fs)

        return raw


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
    parser.add_argument("--ref_rms", help="Reference RMS to normalize to.", type=float,
                        dest="ref_rms", required=False, default=0.1)

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
    loudness_normalizer = LoudnessNormalizer(args.ref_rms)
    loudness_normalizer.process_list(id_list, args.dir_wav, args.dir_out, args.format)


if __name__ == "__main__":
    main()
