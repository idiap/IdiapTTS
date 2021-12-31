#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Bastian Schnell <bastian.schnell@idiap.ch>
#

# Test cases inspired by PyTorch test cases https://github.com/pytorch/pytorch/blob/5b0f40048899e398d286fe7b55f297991f93ba2c/test/test_optim.py


import unittest

import torch
import torch.functional as F
from torch.optim import SGD
import numpy as np

from idiaptts.src.neural_networks.pytorch.ExtendedExponentialLR import ExtendedExponentialLR


class TestExtendedExponentialLR(unittest.TestCase):

    class SchedulerTestNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(1, 1, 1)
            self.conv2 = torch.nn.Conv2d(1, 1, 1)

        def forward(self, x):
            return self.conv2(F.relu(self.conv1(x)))

    def setUp(self):
        self.net = self.SchedulerTestNet()
        self.opt = SGD([{'params': self.net.conv1.parameters()}, {
                       'params': self.net.conv2.parameters(), 'lr': 0.5}], lr=0.05)

    def _run_test(self, schedulers, targets, epochs=10):
        schedulers = [schedulers]
        for epoch in range(epochs):
            for param_group, target in zip(self.opt.param_groups, targets):
                self.assertTrue(np.isclose(target[epoch], param_group['lr'], atol=1e-5, rtol=0),
                    msg='LR is wrong in epoch {}: expected {}, got {}'.format(epoch, target[epoch], param_group['lr']))
            [scheduler.step() for scheduler in schedulers]

    def _run_test_against_closed_form(self, scheduler, closed_form_scheduler, epochs=10):
        targets = []
        for epoch in range(epochs):
            closed_form_scheduler.step(epoch)
            targets.append([group['lr'] for group in closed_form_scheduler.optimizer.param_groups])
        for epoch in range(epochs):
            scheduler.step()
            for i, param_group in enumerate(self.opt.param_groups):
                self.assertTrue(np.isclose(targets[epoch][i], param_group['lr'], atol=1e-5, rtol=0),
                                msg='LR is wrong in epoch {}: expected {}, got {}'.format(epoch, targets[epoch][i],
                                                                                          param_group['lr']))

    def _run_test_lr_is_constant_for_constant_epoch(self, scheduler, epoch=2):
        l = []

        for _ in range(10):
            scheduler.step(epoch)
            l.append(self.opt.param_groups[0]['lr'])
        self.assertEqual(min(l), max(l))

    def test_exponential_lr_is_constant_for_constant_epoch(self):
        scheduler = ExtendedExponentialLR(self.opt, gamma=0.9, min_lr=0.01, warmup_steps=5, decay_steps=2)
        self._run_test_lr_is_constant_for_constant_epoch(scheduler)
        self._run_test_lr_is_constant_for_constant_epoch(scheduler, 8)
        self._run_test_lr_is_constant_for_constant_epoch(scheduler, 50)

    def test_exp_lr(self):
        epochs = 20
        single_targets = list(map(lambda x: max(0.03, x), [0.05] * 5 + [0.05 * (0.9 ** (x/2.0)) for x in range(epochs - 5)]))
        targets = [single_targets, [x * 10 for x in single_targets]]
        scheduler = ExtendedExponentialLR(self.opt, gamma=0.9, min_lr=[0.03, 0.3], warmup_steps=5, decay_steps=2)
        self._run_test(scheduler, targets, epochs)

    def test_closed_form_exp_lr(self):
        closed_form_scheduler = ExtendedExponentialLR(self.opt, gamma=0.9, min_lr=[0.03, 0.25], warmup_steps=5, decay_steps=2)
        self.setUp()
        scheduler = ExtendedExponentialLR(self.opt, gamma=0.9, min_lr=[0.03, 0.25], warmup_steps=5, decay_steps=2)
        self._run_test_against_closed_form(scheduler, closed_form_scheduler, 20)
