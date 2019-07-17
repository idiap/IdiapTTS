#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#

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


def equal_checkpoint(model1_path, model2_path):
    checkpoint1 = torch.load(model1_path)
    checkpoint2 = torch.load(model2_path)

    if not checkpoint1["model_name"] == checkpoint2["model_name"]:
        return False

    if not checkpoint1["epoch"] == checkpoint2["epoch"]:
        return False

    model1 = checkpoint1["model"]
    model2 = checkpoint2["model"]
    if not equal_model(model1, model2):
        return False

    optimiser1 = checkpoint1["optimiser"]
    optimiser2 = checkpoint2["optimiser"]

    if optimiser1 is not None:
        if optimiser2 is not None:
            return equal_iterable(optimiser1.param_groups, optimiser2.param_groups)
        else:
            return False
    return True
