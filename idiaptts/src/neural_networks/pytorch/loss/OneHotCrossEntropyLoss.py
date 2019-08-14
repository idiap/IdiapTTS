#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#

import torch


class OneHotCrossEntropyLoss(torch.nn.CrossEntropyLoss):

    def __init__(self, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='elementwise_mean', shift=1):
        self.shift = shift
        super().__init__(weight, size_average, ignore_index, reduce, reduction)

    def forward(self, input, target):

        # Convert one hot vector to index tensor.
        # (B x C x T) -> (B x T)
        _, targets = target.max(dim=1)

        if self.shift is not None:
            input = input[..., :-self.shift]
            targets = targets[..., self.shift:]

        # input (B x C x T), targets (B x T), loss_full (B x T)
        loss_full = super(OneHotCrossEntropyLoss, self).forward(input, targets)

        # (B x T) -unsqueeze-> (B x T x 1)
        return loss_full.unsqueeze(-1)
