# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase
from unittest.mock import Mock

from mmdet.engine.hooks import MemoryProfilerHook


class TestMemoryProfilerHook(TestCase):

    def test_after_train_iter(self):
        hook = MemoryProfilerHook(2)
        runner = Mock()
        runner.logger = Mock()
        runner.logger.info = Mock()
        hook.after_train_iter(runner, 0)
        runner.logger.info.assert_not_called()
        hook.after_train_iter(runner, 1)
        runner.logger.info.assert_called_once()

    def test_after_val_iter(self):
        hook = MemoryProfilerHook(2)
        runner = Mock()
        runner.logger = Mock()
        runner.logger.info = Mock()
        hook.after_val_iter(runner, 0)
        runner.logger.info.assert_not_called()
        hook.after_val_iter(runner, 1)
        runner.logger.info.assert_called_once()

    def test_after_test_iter(self):
        hook = MemoryProfilerHook(2)
        runner = Mock()
        runner.logger = Mock()
        runner.logger.info = Mock()
        hook.after_test_iter(runner, 0)
        runner.logger.info.assert_not_called()
        hook.after_test_iter(runner, 1)
        runner.logger.info.assert_called_once()
