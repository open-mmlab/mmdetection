# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase
from unittest.mock import Mock

from mmdet.engine.hooks import PipelineSwitchHook


class TestPipelineSwitchHook(TestCase):

    def test_persistent_workers_on(self):
        runner = Mock()
        runner.model = Mock()
        runner.model.module = Mock()
        runner.train_dataloader = Mock()
        runner.train_dataloader.persistent_workers = True
        runner.train_dataloader._DataLoader__initialized = True

        stage2 = [dict(type='RandomResize', scale=(1280, 1280))]

        runner.epoch = 284  # epoch < switch_epoch
        hook = PipelineSwitchHook(switch_epoch=285, switch_pipeline=stage2)
        hook.before_train_epoch(runner)
        self.assertFalse(hook._restart_dataloader)
        self.assertTrue(runner.train_dataloader._DataLoader__initialized)

        runner.epoch = 285  # epoch == switch_epoch
        hook.before_train_epoch(runner)
        self.assertTrue(hook._restart_dataloader)
        self.assertFalse(runner.train_dataloader._DataLoader__initialized)
        self.assertTrue(
            len(runner.train_dataloader.dataset.pipeline.transforms) == 1)

        runner.epoch = 286  # epoch > switch_epoch
        hook.before_train_epoch(runner)
        self.assertTrue(runner.train_dataloader._DataLoader__initialized)
        self.assertTrue(
            len(runner.train_dataloader.dataset.pipeline.transforms) == 1)

    def test_persistent_workers_off(self):
        runner = Mock()
        runner.model = Mock()
        runner.train_dataloader = Mock()
        runner.train_dataloader.persistent_workers = False
        runner.train_dataloader._DataLoader__initialized = True

        stage2 = [dict(type='RandomResize', scale=(1280, 1280))]

        runner.epoch = 284  # epoch < switch_epoch
        hook = PipelineSwitchHook(switch_epoch=285, switch_pipeline=stage2)
        hook.before_train_epoch(runner)
        self.assertFalse(hook._restart_dataloader)
        self.assertTrue(runner.train_dataloader._DataLoader__initialized)

        runner.epoch = 285  # epoch == switch_epoch
        hook.before_train_epoch(runner)
        self.assertFalse(hook._restart_dataloader)
        self.assertTrue(runner.train_dataloader._DataLoader__initialized)
        self.assertTrue(
            len(runner.train_dataloader.dataset.pipeline.transforms) == 1)

        runner.epoch = 286  # epoch > switch_epoch
        hook.before_train_epoch(runner)
        self.assertTrue(runner.train_dataloader._DataLoader__initialized)
        self.assertTrue(
            len(runner.train_dataloader.dataset.pipeline.transforms) == 1)

    def test_initialize_after_switching(self):
        # This simulates the resumption after the switching.
        runner = Mock()
        runner.model = Mock()
        runner.model.module = Mock()
        runner.train_dataloader = Mock()
        runner.train_dataloader.persistent_workers = True
        runner.train_dataloader._DataLoader__initialized = True

        stage2 = [dict(type='RandomResize', scale=(1280, 1280))]

        runner.epoch = 286  # epoch > switch_epoch
        hook = PipelineSwitchHook(switch_epoch=285, switch_pipeline=stage2)
        hook.before_train_epoch(runner)
        self.assertTrue(hook._restart_dataloader)
        self.assertFalse(runner.train_dataloader._DataLoader__initialized)
        self.assertTrue(
            len(runner.train_dataloader.dataset.pipeline.transforms) == 1)
