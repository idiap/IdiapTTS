#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#

import sys
import os
from pydub import AudioSegment

# Local source tree imports.
if not any("IdiapTTS'" in p for p in sys.path):
    parent_dirs = os.path.realpath(__file__).split(os.sep)
    dir_itts = str.join(os.sep, parent_dirs[:parent_dirs.index("IdiapTTS")+1])
    sys.path.append(dir_itts)  # Adds the IdiapTTS folder to the path, required to work on grid.
from misc.utils import makedirs_safe

"""
Function that down-samples a list of audio files.

python down_sampling.py <dir_audio> <dir_out> <file_id_list> <target_sampling_rate>
"""

# Read which files to process.
dir_audio = sys.argv[1]
dir_out = sys.argv[2]
file_id_list = sys.argv[3]
target_sampling_rate = sys.argv[4]

with open(file_id_list) as f:
    id_list = f.readlines()
# Trim entries in-place.
id_list[:] = [s.strip(' \t\n\r') for s in id_list]

for file_id in id_list:
    full_path_in = os.path.join(dir_audio, file_id + ".wav")
    print("Downsample " + full_path_in)
    sound = AudioSegment.from_file(full_path_in)
    sound = sound.set_frame_rate(16000)
    full_path_out = os.path.join(dir_out, file_id + ".wav")
    makedirs_safe(os.path.dirname(full_path_out))
    sound.export(full_path_out, format="wav")
