# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase
from unittest.mock import Mock, patch

import torch.nn as nn

from mmdet.engine.hooks import SyncNormHook


class TestSyncNormHook(TestCase):

    @patch(
        'mmdet.engine.hooks.sync_norm_hook.get_dist_info', return_value=(0, 1))
    def test_before_val_epoch_non_dist(self, mock):
        model = nn.Sequential(
            nn.Conv2d(1, 5, kernel_size=3), nn.BatchNorm2d(5, momentum=0.3),
            nn.Linear(5, 10))
        runner = Mock()
        runner.model = model
        hook = SyncNormHook()
        hook.before_val_epoch(runner)

    @patch(
        'mmdet.engine.hooks.sync_norm_hook.get_dist_info', return_value=(0, 2))
    def test_before_val_epoch_dist(self, mock):
        model = nn.Sequential(
            nn.Conv2d(1, 5, kernel_size=3), nn.BatchNorm2d(5, momentum=0.3),
            nn.Linear(5, 10))
        runner = Mock()
        runner.model = model
        hook = SyncNormHook()
        hook.before_val_epoch(runner)

    @patch(
        'mmdet.engine.hooks.sync_norm_hook.get_dist_info', return_value=(0, 2))
    def test_before_val_epoch_dist_no_norm(self, mock):
        model = nn.Sequential(nn.Conv2d(1, 5, kernel_size=3), nn.Linear(5, 10))
        runner = Mock()
        runner.model = model
        hook = SyncNormHook()
        hook.before_val_epoch(runner)
