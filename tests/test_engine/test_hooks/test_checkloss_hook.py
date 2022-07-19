# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase
from unittest.mock import Mock

import torch

from mmdet.engine.hooks import CheckInvalidLossHook


class TestCheckInvalidLossHook(TestCase):

    def test_after_train_iter(self):
        n = 50
        hook = CheckInvalidLossHook(n)
        runner = Mock()
        runner.logger = Mock()
        runner.logger.info = Mock()

        # Test `after_train_iter` function within the n iteration.
        runner.iter = 10
        outputs = dict(loss=torch.LongTensor([2]))
        hook.after_train_iter(runner, 10, outputs=outputs)
        outputs = dict(loss=torch.tensor(float('nan')))
        hook.after_train_iter(runner, 10, outputs=outputs)
        outputs = dict(loss=torch.tensor(float('inf')))
        hook.after_train_iter(runner, 10, outputs=outputs)

        # Test `after_train_iter` at the n iteration.
        runner.iter = n - 1
        outputs = dict(loss=torch.LongTensor([2]))
        hook.after_train_iter(runner, n - 1, outputs=outputs)
        outputs = dict(loss=torch.tensor(float('nan')))
        with self.assertRaises(AssertionError):
            hook.after_train_iter(runner, n - 1, outputs=outputs)
        outputs = dict(loss=torch.tensor(float('inf')))
        with self.assertRaises(AssertionError):
            hook.after_train_iter(runner, n - 1, outputs=outputs)
