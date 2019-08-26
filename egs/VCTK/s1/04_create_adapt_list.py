#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#

"""Module description:

"""

# System imports.
import logging
import sys
import os
import glob
import random
import multiprocessing

# Third-party imports.

# Local source tree imports.
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd()))))  # Adds the ITTS folder to the path.

create_utterance_splits = False  # If True, creates *_train, *_test, and *_val sets where utterance ids are disjoint.

utts_range = 465
seed = 1234
test_set_size = 226
val_set_size = 226
voices = ["English"]  # ["demo", "half", "full", "English"]
dir_data = os.path.join(os.path.realpath(os.path.dirname(__file__)), "database")

speaker_english = ["p225", "p226", "p227", "p228", "p229", "p230", "p231", "p232", "p233", "p236", "p239", "p240", "p243", "p244", "p250", "p254", "p256", "p257", "p258", "p259", "p267", "p268", "p269", "p270", "p273", "p274", "p276", "p277", "p278", "p279", "p282", "p286", "p287"]
# awk '$4 == "English" {printf "p%s|",$1}' speaker-info.txt  # Print all speakers with English accent.

speaker_adapt = dict()
speaker_adapt["demo"] = ["p269"]
speaker_adapt["full"] = ["p269", "p276", "p277", "p278", "p279", "p281"]
speaker_adapt["half"] = speaker_adapt["full"]
speaker_adapt["English"] = ["p276", "p277", "p278", "p279"]

for key, value in speaker_adapt.items():
    print("Speakers for adaptation {}: {}".format(key, value))

# if create_utterance_splits:
#     random.seed(seed)
#     nums = random.sample(range(utts_range), test_set_size + val_set_size)
#     nums_test = sorted(nums[:test_set_size])
#     nums_test = [str(num).zfill(3) for num in nums_test]
#     nums_val = sorted(nums[test_set_size:])
#     nums_val = [str(num).zfill(3) for num in nums_val]
#
#     print("Test utterance ids: {}".format(nums_test))
#     print("Validation utterance ids: {}".format(nums_val))
#
#     for voice in voices:
#         path_file_id_list = os.path.join(dir_data, "file_id_list_{}.txt".format(voice))
#         path_file_id_list_train = os.path.join(dir_data, "file_id_list_{}_train.txt".format(voice))
#         path_file_id_list_test = os.path.join(dir_data, "file_id_list_{}_test.txt".format(voice))
#         path_file_id_list_val = os.path.join(dir_data, "file_id_list_{}_val.txt".format(voice))
#         path_file_id_list_adapt_train = os.path.join(dir_data, "file_id_list_{}_adapt_train.txt".format(voice))
#         path_file_id_list_adapt_test = os.path.join(dir_data, "file_id_list_{}_adapt_test.txt".format(voice))
#         path_file_id_list_adapt_val = os.path.join(dir_data, "file_id_list_{}_adapt_val.txt".format(voice))
#         with open(path_file_id_list) as file_id_list,\
#              open(path_file_id_list_train, "w") as file_id_list_train,\
#              open(path_file_id_list_test, "w") as file_id_list_test,\
#              open(path_file_id_list_val, "w") as file_id_list_val,\
#              open(path_file_id_list_adapt_train, "w") as file_id_list_adapt_train,\
#              open(path_file_id_list_adapt_test, "w") as file_id_list_adapt_test,\
#              open(path_file_id_list_adapt_val, "w") as file_id_list_adapt_val:
#
#             for line in file_id_list:
#                 speaker_id, utt_id = line.strip().split('/')[1].split('_')
#
#                 if speaker_id == "p236" and utt_id in ["002", "006", "063", "066", "071", "072", "073", "083", "086",
#                                                        "087", "088", "090", "096", "101", "102", "102", "103", "104"]:
#                     # The recordings listed here have a lot of noise and the extraction of VUV fails so we exclude them.
#                     continue
#                 elif speaker_id in speaker_adapt[voice]:
#                     if utt_id in nums_test:
#                         file_id_list_adapt_test.write(line)
#                     elif utt_id in nums_val:
#                         file_id_list_adapt_val.write(line)
#                     else:
#                         file_id_list_adapt_train.write(line)
#                 else:
#                     if utt_id in nums_test:
#                         file_id_list_test.write(line)
#                     elif utt_id in nums_val:
#                         file_id_list_val.write(line)
#                     else:
#                         file_id_list_train.write(line)
# else:

num_utts_splits = [10, 380]
num_utts_listening_test = 8

for voice in voices:
    # Open file id list and create a dictionary with [speaker_id][utt_id], save maximum utterance id.
    # Already create training set and update file id list itself (remove noisy samples).
    path_file_id_list = os.path.join(dir_data, "file_id_list_{}.txt".format(voice))
    path_file_id_list_train = os.path.join(dir_data, "file_id_list_{}_train.txt".format(voice))
    utts_dict = dict()
    max_utt_id = 0
    file_id_list_new = list()
    with open(path_file_id_list) as file_id_list,\
         open(path_file_id_list_train, "w") as file_id_list_train:
        for line in file_id_list:
            speaker_id, utt_id = line.strip().split('/')[1].split('_')
            max_utt_id = max_utt_id if max_utt_id >= int(utt_id) else int(utt_id)

            # The recordings listed here have a lot of noise and the extraction of VUV fails so we exclude them.
            if not (speaker_id == "p236" and utt_id in ["002", "006", "063", "066", "071", "072", "073", "083", "086",
                                                        "087", "088", "090", "096", "101", "102", "102", "103", "104"]):
                if speaker_id in speaker_adapt[voice]:
                    if speaker_id not in utts_dict:
                        utts_dict[speaker_id] = {int(utt_id): line}
                    else:
                        utts_dict[speaker_id][int(utt_id)] = line
                else:
                    file_id_list_train.write(line)
                file_id_list_new.append(line)
    # Write modified file id list back (this one excludes the noisy recordings).
    with open(path_file_id_list, "w") as file_id_list:
        assert(len(file_id_list_new) > 0)
        file_id_list.writelines(file_id_list_new)

    possible_training_utts = list()
    for utt_id in range(0, max_utt_id):
        is_valid = True
        for speaker_id, speaker_utts_dict in utts_dict.items():
            if utt_id not in speaker_utts_dict:
                is_valid = False
                break  # Breaks out of dict loop so that next utt_id is tried.
        if is_valid:
            # Utterance id is available for all adaptation speakers.
            possible_training_utts.append(utt_id)

    # Shuffle list of possible training ids.
    random.seed(seed)
    random.shuffle(possible_training_utts)

    # Select num_utts_splits for training.
    for num_utts_train in num_utts_splits:
        num_utts_test = (len(possible_training_utts) - num_utts_train) // 2
        utts_listening_test = possible_training_utts[-num_utts_listening_test:]
        utts_test = sorted(possible_training_utts[-num_utts_test:])
        utts_train = sorted(possible_training_utts[-num_utts_test - num_utts_train:-num_utts_test])

        print("Train utterance for {} ids ({}): {}".format(num_utts_train, len(utts_train), utts_train))
        print("Test utterance for {} ids ({}): {}".format(num_utts_train, len(utts_test), utts_test))
        print("Listening test utterance for {} ids ({}): {}".format(num_utts_train, len(utts_listening_test), utts_listening_test))

        path_file_id_list_adapt_train = os.path.join(dir_data, "file_id_list_{}_adapt{}_train.txt".format(voice, num_utts_train))
        path_file_id_list_adapt_val = os.path.join(dir_data, "file_id_list_{}_adapt{}_val.txt".format(voice, num_utts_train))
        path_file_id_list_adapt_test = os.path.join(dir_data, "file_id_list_{}_adapt{}_test.txt".format(voice, num_utts_train))
        path_file_id_list_adapt_listening_test = os.path.join(dir_data, "file_id_list_{}_adapt{}_listening_test.txt".format(voice, num_utts_train))

        with open(path_file_id_list_adapt_train, "w") as file_id_list_adapt_train, \
             open(path_file_id_list_adapt_val, "w") as file_id_list_adapt_val, \
             open(path_file_id_list_adapt_test, "w") as file_id_list_adapt_test, \
             open(path_file_id_list_adapt_listening_test, "w") as file_id_list_adapt_listening_test:

            for speaker_id, speaker_utts_dict in utts_dict.items():
                for utt_id, line in speaker_utts_dict.items():
                    if utt_id in utts_train:
                        file_id_list_adapt_train.write(line)
                    elif utt_id in utts_test:
                        file_id_list_adapt_test.write(line)
                        if speaker_id in ["p276", "p278"] and utt_id in utts_listening_test:
                            file_id_list_adapt_listening_test.write(line)
                    else:
                        file_id_list_adapt_val.write(line)
