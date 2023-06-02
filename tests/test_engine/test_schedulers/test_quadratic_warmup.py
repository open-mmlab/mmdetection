# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
import torch.nn.functional as F
import torch.optim as optim
from mmengine.optim.scheduler import _ParamScheduler
from mmengine.testing import assert_allclose

from mmdet.engine.schedulers import (QuadraticWarmupLR,
                                     QuadraticWarmupMomentum,
                                     QuadraticWarmupParamScheduler)


class ToyModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 1, 1)
        self.conv2 = torch.nn.Conv2d(1, 1, 1)

    def forward(self, x):
        return self.conv2(F.relu(self.conv1(x)))


class TestQuadraticWarmupScheduler(TestCase):

    def setUp(self):
        """Setup the model and optimizer which are used in every test method.

        TestCase calls functions in this order: setUp() -> testMethod() ->
        tearDown() -> cleanUp()
        """
        self.model = ToyModel()
        self.optimizer = optim.SGD(
            self.model.parameters(), lr=0.05, momentum=0.01, weight_decay=5e-4)

    def _test_scheduler_value(self,
                              schedulers,
                              targets,
                              epochs=10,
                              param_name='lr'):
        if isinstance(schedulers, _ParamScheduler):
            schedulers = [schedulers]
        for epoch in range(epochs):
            for param_group, target in zip(self.optimizer.param_groups,
                                           targets):
                print(param_group[param_name])
                assert_allclose(
                    target[epoch],
                    param_group[param_name],
                    msg='{} is wrong in epoch {}: expected {}, got {}'.format(
                        param_name, epoch, target[epoch],
                        param_group[param_name]),
                    atol=1e-5,
                    rtol=0)
            [scheduler.step() for scheduler in schedulers]

    def test_quadratic_warmup_scheduler(self):
        with self.assertRaises(ValueError):
            QuadraticWarmupParamScheduler(self.optimizer, param_name='lr')
        epochs = 10
        iters = 5
        warmup_factor = [pow((i + 1) / float(iters), 2) for i in range(iters)]
        single_targets = [x * 0.05 for x in warmup_factor] + [0.05] * (
            epochs - iters)
        targets = [single_targets, [x * epochs for x in single_targets]]
        scheduler = QuadraticWarmupParamScheduler(
            self.optimizer, param_name='lr', end=iters)
        self._test_scheduler_value(scheduler, targets, epochs)

    def test_quadratic_warmup_scheduler_convert_iterbased(self):
        epochs = 10
        end = 5
        epoch_length = 11

        iters = end * epoch_length
        warmup_factor = [pow((i + 1) / float(iters), 2) for i in range(iters)]
        single_targets = [x * 0.05 for x in warmup_factor] + [0.05] * (
            epochs * epoch_length - iters)
        targets = [single_targets, [x * epochs for x in single_targets]]
        scheduler = QuadraticWarmupParamScheduler.build_iter_from_epoch(
            self.optimizer,
            param_name='lr',
            end=end,
            epoch_length=epoch_length)
        self._test_scheduler_value(scheduler, targets, epochs * epoch_length)

    def test_quadratic_warmup_lr(self):
        epochs = 10
        iters = 5
        warmup_factor = [pow((i + 1) / float(iters), 2) for i in range(iters)]
        single_targets = [x * 0.05 for x in warmup_factor] + [0.05] * (
            epochs - iters)
        targets = [single_targets, [x * epochs for x in single_targets]]
        scheduler = QuadraticWarmupLR(self.optimizer, end=iters)
        self._test_scheduler_value(scheduler, targets, epochs)

    def test_quadratic_warmup_momentum(self):
        epochs = 10
        iters = 5
        warmup_factor = [pow((i + 1) / float(iters), 2) for i in range(iters)]
        single_targets = [x * 0.01 for x in warmup_factor] + [0.01] * (
            epochs - iters)
        targets = [single_targets, [x * epochs for x in single_targets]]
        scheduler = QuadraticWarmupMomentum(self.optimizer, end=iters)
        self._test_scheduler_value(
            scheduler, targets, epochs, param_name='momentum')
