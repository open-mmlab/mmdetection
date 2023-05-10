# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase
from unittest.mock import Mock, patch

from mmdet.engine.hooks import YOLOXModeSwitchHook


class TestYOLOXModeSwitchHook(TestCase):

    @patch('mmdet.engine.hooks.yolox_mode_switch_hook.is_model_wrapper')
    def test_is_model_wrapper_and_persistent_workers_on(
            self, mock_is_model_wrapper):
        mock_is_model_wrapper.return_value = True
        runner = Mock()
        runner.model = Mock()
        runner.model.module = Mock()
        runner.model.module.detector.bbox_head.use_l1 = False
        runner.train_dataloader = Mock()
        runner.train_dataloader.persistent_workers = True
        runner.train_dataloader._DataLoader__initialized = True
        runner.epoch = 284
        runner.max_epochs = 300

        hook = YOLOXModeSwitchHook(num_last_epochs=15)
        hook.before_train_epoch(runner)
        self.assertTrue(hook._restart_dataloader)
        self.assertTrue(runner.model.module.detector.bbox_head.use_l1)
        self.assertFalse(runner.train_dataloader._DataLoader__initialized)

        runner.epoch = 285
        hook.before_train_epoch(runner)
        self.assertTrue(runner.train_dataloader._DataLoader__initialized)

    def test_not_model_wrapper_and_persistent_workers_off(self):
        runner = Mock()
        runner.model = Mock()
        runner.model.detector.bbox_head.use_l1 = False
        runner.train_dataloader = Mock()
        runner.train_dataloader.persistent_workers = False
        runner.train_dataloader._DataLoader__initialized = True
        runner.epoch = 284
        runner.max_epochs = 300

        hook = YOLOXModeSwitchHook(num_last_epochs=15)
        hook.before_train_epoch(runner)
        self.assertFalse(hook._restart_dataloader)
        self.assertTrue(runner.model.detector.bbox_head.use_l1)
        self.assertTrue(runner.train_dataloader._DataLoader__initialized)

        runner.epoch = 285
        hook.before_train_epoch(runner)
        self.assertFalse(hook._restart_dataloader)
        self.assertTrue(runner.train_dataloader._DataLoader__initialized)

    @patch('mmdet.engine.hooks.yolox_mode_switch_hook.is_model_wrapper')
    def test_initialize_after_switching(self, mock_is_model_wrapper):
        # This simulates the resumption after the switching.
        mock_is_model_wrapper.return_value = True
        runner = Mock()
        runner.model = Mock()
        runner.model.module = Mock()
        runner.model.module.bbox_head.use_l1 = False
        runner.train_dataloader = Mock()
        runner.train_dataloader.persistent_workers = True
        runner.train_dataloader._DataLoader__initialized = True
        runner.epoch = 285
        runner.max_epochs = 300

        # epoch + 1 > max_epochs - num_last_epochs .
        hook = YOLOXModeSwitchHook(num_last_epochs=15)
        hook.before_train_epoch(runner)
        self.assertTrue(hook._restart_dataloader)
        self.assertTrue(runner.model.module.detector.bbox_head.use_l1)
        self.assertFalse(runner.train_dataloader._DataLoader__initialized)
