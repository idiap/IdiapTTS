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
import argparse

# Third-party imports.
import soundfile
import matlab
import matlab.engine

# Local source tree imports.
from idiaptts.misc.utils import makedirs_safe


class SingleChannelNoiseReduction(object):
    logger = logging.getLogger(__name__)

    def __init__(self):
        self.eng = matlab.engine.connect_matlab()  # Connects to found shared session or creates a new shared session.
        self.eng.addpath(os.path.join(os.environ["IDIAPTTS_ROOT"], "scripts", "noise_reduction"))

    @staticmethod
    def nparray_to_matlab(x):
        return matlab.double(x.tolist())

    def process_list(self, id_list, dir_audio, dir_out, extension="wav"):

        for file_id in id_list:
            self.process_file(file_id + "." + extension, dir_audio, dir_out)

    def process_file(self, file, dir_audio, dir_out):

        raw, fs = soundfile.read(os.path.join(dir_audio, file))

        data_noisy_matlab = self.nparray_to_matlab(raw)
        data_noisy_matlab = self.eng.transpose(data_noisy_matlab)

        enhanced = self.eng.runme(data_noisy_matlab, fs)

        out_file = os.path.join(dir_out, file)
        makedirs_safe(os.path.dirname(out_file))
        soundfile.write(out_file, enhanced, samplerate=fs)

        return enhanced


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
    loudness_normalizer = SingleChannelNoiseReduction()
    loudness_normalizer.process_list(id_list, args.dir_wav, args.dir_out, args.format)


if __name__ == "__main__":
    main()
