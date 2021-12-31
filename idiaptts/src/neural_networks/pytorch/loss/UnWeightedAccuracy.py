#
# Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#

import numpy as np
from sklearn.metrics import confusion_matrix
import torch


class UnWeightedAccuracy(torch.nn.modules.loss._Loss):

    def __init__(self, num_per_class: torch.Tensor, reduction: str = 'none'):
        self._num_classes = len(num_per_class)
        self._num_per_class = num_per_class
        self._num_total = num_per_class.sum().float()
        self._num_per_class[self._num_per_class == 0] = 1.0  # Prevent NaNs.
        super().__init__()

    def forward(self, input_, target):
        class_pred = input_.argmax(dim=-1)

        conf_matrix = confusion_matrix(target.cpu(), class_pred.cpu(),
                                       labels=range(self._num_classes))
        conf_matrix_norm = conf_matrix / self._num_per_class

        weighted_acc = (class_pred == target).sum() / self._num_total
        unweighted_acc = np.sum(np.diag(conf_matrix_norm)) / self._num_classes
        unweighted_acc = torch.tensor([unweighted_acc], dtype=torch.float32, device=input_.device)
        # print(conf_matrix)
        # print(num_per_class)
        # print(weighted_acc)
        # print(unweighted_acc)

        return -(weighted_acc + unweighted_acc)
