#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#

from datetime import datetime, timezone
from functools import reduce
import logging
import math
import operator as op
import os
import subprocess

import git
import numpy as np
import torch
import scipy
from scipy.stats import norm


def makedirs_safe(directory):
    """Make directory operation which does not fail if directory already exists."""
    try:
        os.makedirs(directory)
    except OSError:
        if not os.path.isdir(directory):
            raise


def file_len(fname):
    """Count the number of lines in a file."""
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def interpolate_lin(data):
    """
    Create continuous f0/lf0 and vuv from uncontinuous f0/lf0.
    Code from merlin/src/frontend/acoustic_normalisation.
    https://github.com/CSTR-Edinburgh/merlin

    :return:  interpolated_f0, vuv_vector
    """

    data = np.reshape(np.copy(data), (data.size, 1))

    vuv_vector = np.zeros((data.size, 1))
    vuv_vector[data > 0.0] = 1.0
    vuv_vector[data <= 0.0] = 0.0

    ip_data = data

    frame_number = data.size
    last_value = 0.0
    for i in range(frame_number):
        if data[i] <= 0.0:
            j = i + 1
            # Jump over succeeding zero values.
            for j in range(i + 1, frame_number):
                if data[j] > 0.0:
                    break
            # Check if end of data is reached.
            if j < frame_number - 1:
                # If not and a valid last data point exists, interpolate linear between the two non-zero values.
                if last_value > 0.0:
                    step = (data[j] - data[i - 1]) / float(j - i)
                    for k in range(i, j):
                        ip_data[k] = data[i - 1] + step * (k - i + 1)
                else:
                    # If no valid last data point exists (zeros in the beginning) fill it with first valid value
                    for k in range(i, j):
                        ip_data[k] = data[j]
            else:
                # If end of data is reached (zeros in the end) fill it with last valid value.
                for k in range(i, frame_number):
                    ip_data[k] = last_value
        else:
            # Data is valid, so use it as interpolated data and save it as last valid value.
            ip_data[i] = data[i]
            last_value = data[i]

    return ip_data, vuv_vector


def sample_linearly(sample, in_to_out_multiplier, dtype=np.float32):

    if in_to_out_multiplier > 1:
        x = np.arange(0.0, len(sample))
        f = scipy.interpolate.interp1d(x, sample, axis=0)
        x_new = np.linspace(start=0.0, stop=len(sample) - 1, num=int(in_to_out_multiplier) * len(sample))
        sample_sampled = np.array(f(x_new), dtype=dtype)
        return sample_sampled
    elif in_to_out_multiplier < 1:
        raise NotImplementedError()  # TODO: Implement down-sampling.
    else:
        return sample


def compute_deltas(labels):
    labels_deltas = np.gradient(labels, axis=0).astype(dtype=np.float32)
    return labels_deltas


def surround_with_norm_dist(label, window_size=5, std_dev=1.0, mean=0.0, threshold=0.2):
    """Surrounds each non-zero value by a normal distribution."""
    if window_size % 2 == 0:
        window_size += 1

    # # Convert to beta dist atoms.
    # window_size = window_size + 2
    # a_b = 3
    # beta_dist = beta(a_b, a_b, scale=1.0, loc=0.0)  # Scale determines width of beta distribution.
    # beta_max = beta_dist.pdf(0.5)
    # beta_coefs = beta_dist.pdf(np.linspace(0, 1, window_size)) / beta_max

    # Convert to norm dist atoms.
    threshold_x = abs((mean + math.sqrt(-math.log(threshold) * 2 * std_dev ** 2 - mean ** 2)))
    norm_dist = norm(loc=mean, scale=std_dev)
    norm_max = norm_dist.pdf(mean)
    norm_coefs = norm_dist.pdf(np.linspace(-threshold_x, threshold_x, window_size)) / norm_max

    # Apply distribution.
    dist_coefs = norm_coefs.reshape(-1, 1)
    dist_label = np.zeros(label.shape)

    # Get indices of atoms (non-zero entries).
    atoms_pos, = np.nonzero(label[:, 0] != 0)
    # print id_name + ": " + str(atoms_pos)

    # Surround each atom with a distribution.
    for pos in atoms_pos:
        start = pos - int(window_size / 2)
        dist_start = 0
        dist_end = window_size
        if start < 0:
            dist_start = abs(start)
            start = 0
        end = pos + int(window_size / 2)
        if end >= len(label):
            dist_end = window_size - (end - len(label) + 1)
            end = len(label) - 1
        dist_label[start:end + 1] += np.dot(dist_coefs[dist_start:dist_end], label[pos].reshape(1, -1))
        # for local_i, i in enumerate(xrange(max(0, pos - int(window_size/2)), min(len(label)-1, pos + int(window_size/2)))):
        #     dist_label[i] += dist.pdf(float(local_i + 1)/float(window_size)) / norm_max * label[pos] # Scale beta distribution by atom amp.
    return dist_label


def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    # return (torch.cuda.memory_allocated() / (1024**2))

    try:
        result = subprocess.check_output(
            [
                'nvidia-smi', '--query-gpu=memory.used',
                '--format=csv,nounits,noheader'
            ], encoding='utf-8')
    except subprocess.CalledProcessError as e:
        return "not availabe"

    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map


def ncr(n, r):
    r = min(r, n - r)
    numer = reduce(op.mul, range(n, n - r, -1), 1)
    denom = reduce(op.mul, range(1, r + 1), 1)
    return numer / denom


def parse_int_set(nputstr=""):
    """
    Parse a set of selected values from a comma separated list of integers.
    Code from http://thoughtsbyclayg.blogspot.com/2008/10/parsing-list-of-numbers-in-python.html

    Example:
    1-4,6 returns 1,2,3,4,6

    :param nputstr:         Comma separated list of integers (can contain ranges).
    :return:                A list of integers.
    """
    selection = set()
    invalid = set()
    # tokens are comma seperated values
    tokens = [x.strip() for x in nputstr.split(',')]
    for i in tokens:
        try:
            # typically tokens are plain old integers
            selection.add(int(i))
        except:
            # if not, then it might be a range
            try:
                token = [int(k.strip()) for k in i.split('-')]
                if len(token) > 1:
                    token.sort()
                    # we have items seperated by a dash
                    # try to build a valid range
                    first = token[0]
                    last = token[len(token)-1]
                    for x in range(first, last+1):
                        selection.add(x)
            except:
               # not an int and not a range...
               invalid.add(i)
    # Report invalid tokens before returning valid selection
    if len(invalid) > 0:
        print("Invalid set: " + str(invalid))
    return selection


def select_skip(iterable, select, skip, start_index=0):
    """
    Select a region of elements from an interable every skip steps.

    :param iterable:             Iterable data representation.
    :param select:               Size of the region which should be selected.
    :param skip:                 Number of elements skipped between regions.
    :param start_index:          Start index of first region.
    :return:                     List of elements in selected regions.
    """
    return [x for i, x in enumerate(iterable) if (i - start_index) % (select+skip) < select]


def local_modification_time(file_path):
    utc_time = datetime.fromtimestamp(os.path.getmtime(file_path), timezone.utc)
    local_time = utc_time.astimezone()
    mod_time = local_time.strftime('%Y-%m-%d %H:%M:%S')
    return mod_time


def log_git_hash():
    try:
        repo = git.Repo(search_parent_directories=True)
        logging.info("Git: {} at {}".format(repo.git_dir, repo.head.object.hexsha))
    except git.exc.InvalidGitRepositoryError:
        pass
    try:
        framework_repo = git.Repo(path=os.environ['IDIAPTTS_ROOT'], search_parent_directories=True)
        logging.info("IdiapTTS framework git: {} at {}"
                         .format(framework_repo.git_dir, framework_repo.head.object.hexsha))
    except git.exc.InvalidGitRepositoryError:
        pass


def pretty_print_nested(value, htchar='\t', lfchar='\n', indent=0,
                        max_array_elements=15):
    nlch = lfchar + htchar * (indent + 1)
    if type(value) is dict:
        if len(value) == 0:
            return '{}'
        items = [
            nlch + repr(key) + ': ' + pretty_print_nested(
                value[key], htchar, lfchar, indent + 1, max_array_elements)
            for key in sorted(value)
        ]
        return '{%s}' % (','.join(items) + lfchar + htchar * indent)
    elif type(value) is list:
        if len(value) == 0:
            return '[]'
        elif len(value) > 1:
            items = [
                nlch + pretty_print_nested(
                    item, htchar, lfchar, indent + 1, max_array_elements)
                for item in value
            ]
            return '[%s]' % (','.join(items) + lfchar + htchar * indent)
        else:
            return '[' + pretty_print_nested(
                value[0], htchar, lfchar, indent + 1, max_array_elements) + ']'
    elif type(value) is tuple:
        items = [
            nlch + pretty_print_nested(
                item, htchar, lfchar, indent + 1, max_array_elements)
            for item in value
        ]
        return '(%s)' % (','.join(items) + lfchar + htchar * indent)
    elif isinstance(value, np.ndarray) and value.size > max_array_elements:
        return '{} {}'.format(type(value), value.shape)
    elif isinstance(value, torch.Tensor) and value.numel() > max_array_elements:
        return '{} {}'.format(type(value), value.shape)
    else:
        return repr(value)


def pretty_print_decimal_places(lr):
    return str(np.format_float_positional(lr).split('.')[1])
