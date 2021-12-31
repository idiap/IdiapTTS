#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#
import os
import filecmp

import torch


def equal_iterable(item1, item2):
    # if torch.is_tensor(item1) and torch.is_tensor(item2):
    #     return bool(item1.ne(item2).sum() == 0)
    # if isinstance(item1, numpy.ndarray) and isinstance(item2, numpy.ndarray):
    #     return bool(numpy.equal(item1, item2).all())

    if isinstance(item1, str) and isinstance(item2, str):
        return item1 == item2

    try:
        if isinstance(item1, dict):
            iter1 = item1.values()
            if isinstance(item2, dict):
                iter2 = item2.values()
            else:
                return False
        else:
            iter1 = iter(item1)
            iter2 = iter(item2)

        equal = True
        for value1, value2 in zip(iter1, iter2):
            equal &= equal_iterable(value1, value2)
        return equal
    except TypeError as te:
        equal = (item1 == item2)
        # This is basically a check for torch and numpy tensors/arrays.
        try:
            iter(equal)
            return any(equal)
        except TypeError as te2:
            return equal


def equal_model(model1, model2):
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            return False
        if p1.grad is not None:
            if p2.grad is not None:
                if p1.grad.ne(p2.grad).sum() > 0:
                    return False
            else:
                return False
        elif p2.grad is not None:
            return False
    return True


def equal_checkpoint(model1_path, model1_suffix, model2_path, model2_suffix):

    model1_params_path = os.path.join(model1_path, "params_" + model1_suffix)
    model2_params_path = os.path.join(model2_path, "params_" + model2_suffix)
    model1_optimiser_path = os.path.join(model1_path, "optimiser_" + model1_suffix)
    model2_optimiser_path = os.path.join(model2_path, "optimiser_" + model2_suffix)
    model1_config_path = os.path.join(model1_path, "config.json")
    model2_config_path = os.path.join(model2_path, "config.json")

    # Try binary test first.
    return filecmp.cmp(os.path.join(model1_params_path), os.path.join(model2_params_path), shallow=False) \
        and filecmp.cmp(os.path.join(model1_optimiser_path), os.path.join(model2_optimiser_path), shallow=False) \
        and (model1_path == model2_path or filecmp.cmp(model1_config_path, model2_config_path, shallow=False))

    # # When binary test fails check details manually. The only difference might be the save epoch.
    # checkpoint1 = torch.load(model1_path)
    # checkpoint2 = torch.load(model2_path)

    # if not checkpoint1["model_name"] == checkpoint2["model_name"]:
    #     return False

    # if not checkpoint1["epoch"] == checkpoint2["epoch"]:
    #     return False

    # try:
    #     model1 = checkpoint1["model"]
    #     model2 = checkpoint2["model"]
    #     if not equal_model(model1, model2):
    #         return False
    # except KeyError:
    #     state_dict1 = checkpoint1["model_state_dict"]
    #     state_dict2 = checkpoint2["model_state_dict"]
    #     if not state_dict1.keys() == state_dict2.keys():
    #         return False
    #     for key, value in state_dict1.items():
    #         if not (state_dict2[key] == value).all():
    #             return False

    # # Backwards compatibility for fully saved optimisers.
    # try:
    #     optimiser1_state_dict = checkpoint1["optimiser"].state_dict()
    # except KeyError:
    #     optimiser1_state_dict = checkpoint1["optimiser_state_dict"]
    # try:
    #     optimiser2_state_dict = checkpoint2["optimiser"].state_dict()
    # except KeyError:
    #     optimiser2_state_dict = checkpoint2["optimiser_state_dict"]

    # if optimiser1_state_dict is not None:
    #     if optimiser2_state_dict is not None:
    #         return equal_iterable(optimiser1_state_dict["state"], optimiser2_state_dict["state"])
    #     else:
    #         return False
    # return True


def tensor_pad(tensor: torch.Tensor, target_length: int, dim: int,
               mode: str = 'constant', value: float = 0.0):
    ndim = tensor.ndim
    assert dim < ndim, "Cannot pad dim {} of {}-dimensional tensor.".format(
        dim, ndim)
    current_length = tensor.shape[dim]
    assert current_length <= target_length, "Tensor is longer than padding " \
        "length ({} > {}).".format(current_length, target_length)

    dim_padding = [0, target_length - tensor.shape[dim]]
    padding = [0, 0] * dim + dim_padding + [0, 0] * (ndim - dim - 1)

    return torch.nn.functional.pad(tensor, padding, mode=mode, value=value)
