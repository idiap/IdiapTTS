#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#

import sys
import os
import librosa
from shutil import copy
from pydub import AudioSegment

# Local source tree imports.
from idiaptts.misc.utils import makedirs_safe

"""
Function that down-samples a list of audio files.

python down_sampling.py <dir_audio> <dir_out> <file_id_list> <target_sampling_rate>
"""

# Read which files to process.
dir_audio = sys.argv[1]
dir_out = sys.argv[2]
file_id_list = sys.argv[3]
target_sampling_rate = int(sys.argv[4])

with open(file_id_list) as f:
    id_list = f.readlines()
# Trim entries in-place.
id_list[:] = [s.strip(' \t\n\r') for s in id_list]

for file_id in id_list:
    full_path_in = os.path.join(dir_audio, file_id + ".wav")
    full_path_out = os.path.join(dir_out, file_id + ".wav")
    makedirs_safe(os.path.dirname(full_path_out))

    current_sampling_rate = librosa.get_samplerate(full_path_in)
    if current_sampling_rate != target_sampling_rate:
        print("Downsample {} from {} to {}.".format(full_path_in, current_sampling_rate, target_sampling_rate))
        y, s = librosa.load(full_path_in, sr=target_sampling_rate)
        librosa.output.write_wav(full_path_out, y, target_sampling_rate)
    else:
        print("Already at target frame rate, so copy " + full_path_in)
        copy(full_path_in, full_path_out)
    # sound = AudioSegment.from_file(full_path_in)
    # sound = sound.set_frame_rate(target_sampling_rate)
    # sound.export(full_path_out, format="wav")
