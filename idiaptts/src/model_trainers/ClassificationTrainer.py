#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#

import logging
import os
from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

from idiaptts.src.ExtendedHParams import ExtendedHParams
from idiaptts.src.model_trainers.ModularTrainer import ModularTrainer


class ClassificationTrainer(ModularTrainer):

    @staticmethod
    def create_hparams(hparams_string: os.PathLike = None,
                       verbose: bool = False):
        hparams = ModularTrainer.create_hparams(hparams_string=hparams_string,
                                                verbose=verbose)
        hparams.add_hparams(
            class_pred_name="class_pred",
            class_true_name="class_true",
            num_classes=-1,
            class_names=None
        )
        return hparams

    def compute_score(self, data: Dict[str, object], output: Dict[str, object],
                      hparams: ExtendedHParams):
        # TODO: Reuse UnWeightedAccuracy class here.
        class_pred = []
        class_true = []
        for id_, out_dict in output.items():
            class_pred.append(out_dict[hparams.class_pred_name].argmax())
            class_true.append(out_dict[hparams.class_true_name].squeeze())
        class_pred = np.stack(class_pred)
        class_true = np.stack(class_true)

        conf_matrix = confusion_matrix(class_true, class_pred,
                                       labels=range(hparams.num_classes))
        num_per_class = conf_matrix.sum(axis=1)[:, None]
        num_per_class[num_per_class == 0] = 1.0  # Prevent NaNs.
        conf_matrix_norm = conf_matrix / num_per_class
        W_acc = accuracy_score(class_true, class_pred)
        U_acc = np.sum(np.diag(conf_matrix_norm)) / hparams.num_classes

        self.logger.info("Weighted accuracy {}".format(W_acc))
        self.logger.info("Unweighted accuracy {}".format(U_acc))
        self.logger.info("Confusion matrix\n{}\n{}".format(
            hparams.class_names if hparams.class_names is not None else "",
            conf_matrix_norm))

        return {"W_acc": W_acc, "U_acc": U_acc, "conf_matrix": conf_matrix_norm}
