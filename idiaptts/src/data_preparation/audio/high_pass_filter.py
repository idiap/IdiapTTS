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

# Local source tree imports.
from scipy import signal

from idiaptts.misc.utils import makedirs_safe


class HighPassFilter(object):
    logger = logging.getLogger(__name__)

    def __init__(self, stop_freq_Hz=70, pass_freq_Hz=100, filter_order=1001):
        self.stop_freq_Hz = stop_freq_Hz
        self.pass_freq_Hz = pass_freq_Hz
        self.filter_order = filter_order

    def process_list(self, id_list, dir_audio, dir_out, format="wav"):

        for file_id in id_list:
            self.process_file(file_id + "." + format, dir_audio, dir_out)

    def process_file(self, file, dir_audio, dir_out):

        raw, fs = soundfile.read(os.path.join(dir_audio, file))

        raw = self.highpass_filter(raw, fs)

        out_file = os.path.join(dir_out, file)
        makedirs_safe(os.path.dirname(out_file))
        soundfile.write(out_file, raw, samplerate=fs)

        return raw

    def highpass_filter(self, raw, fs):

        nyquist_rate = fs / 2.
        desired = (0, 0, 1, 1)
        bands = (0, self.stop_freq_Hz, self.pass_freq_Hz, nyquist_rate)
        filter_coefs = signal.firls(self.filter_order, bands, desired, nyq=nyquist_rate)

        filtered_raw = signal.filtfilt(filter_coefs, [1], raw)
        return filtered_raw


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
    parser.add_argument("--stop_freq_Hz", help="Frequencies below are blocked.", type=float,
                        dest="stop_freq_Hz", required=False, default=70)
    parser.add_argument("--pass_freq_Hz", help="Filter gain raises between stop_freq_Hz and pass_freq_Hz from 0 to 1.",
                        type=float, dest="pass_freq_Hz", required=False, default=100)
    parser.add_argument("--filter_order", help="Size of the filter window in raw signal.", type=int,
                        dest="filter_order", required=False, default=1001)

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
    high_pass_filter = HighPassFilter(args.stop_freq_Hz, args.pass_freq_Hz, args.filter_order)
    high_pass_filter.process_list(id_list, args.dir_wav, args.dir_out, args.format)


if __name__ == "__main__":
    main()
