import logging
import shutil
import sys
import tempfile
from unittest.mock import MagicMock, call

import pytest
import torch
import torch.nn as nn
from mmcv.runner import IterTimerHook, PaviLoggerHook, build_runner
from torch.utils.data import DataLoader

from mmdet.core.hook import YOLOXLrUpdaterHook


def _build_demo_runner_without_hook(runner_type='EpochBasedRunner',
                                    max_epochs=1,
                                    max_iters=None,
                                    multi_optimziers=False):

    class Model(nn.Module):

        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(2, 1)
            self.conv = nn.Conv2d(3, 3, 3)

        def forward(self, x):
            return self.linear(x)

        def train_step(self, x, optimizer, **kwargs):
            return dict(loss=self(x))

        def val_step(self, x, optimizer, **kwargs):
            return dict(loss=self(x))

    model = Model()

    if multi_optimziers:
        optimizer = {
            'model1':
            torch.optim.SGD(model.linear.parameters(), lr=0.02, momentum=0.95),
            'model2':
            torch.optim.SGD(model.conv.parameters(), lr=0.01, momentum=0.9),
        }
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.02, momentum=0.95)

    tmp_dir = tempfile.mkdtemp()
    runner = build_runner(
        dict(type=runner_type),
        default_args=dict(
            model=model,
            work_dir=tmp_dir,
            optimizer=optimizer,
            logger=logging.getLogger(),
            max_epochs=max_epochs,
            max_iters=max_iters))
    return runner


def _build_demo_runner(runner_type='EpochBasedRunner',
                       max_epochs=1,
                       max_iters=None,
                       multi_optimziers=False):
    log_config = dict(
        interval=1, hooks=[
            dict(type='TextLoggerHook'),
        ])

    runner = _build_demo_runner_without_hook(runner_type, max_epochs,
                                             max_iters, multi_optimziers)

    runner.register_checkpoint_hook(dict(interval=1))
    runner.register_logger_hooks(log_config)
    return runner


@pytest.mark.parametrize('multi_optimziers', (True, False))
def test_yolox_lrupdater_hook(multi_optimziers):
    """xdoctest -m tests/test_hooks.py test_cosine_runner_hook."""
    # Only used to prevent program errors
    YOLOXLrUpdaterHook(0, 0, min_lr_ratio=0)

    sys.modules['pavi'] = MagicMock()
    loader = DataLoader(torch.ones((10, 2)))
    runner = _build_demo_runner(multi_optimziers=multi_optimziers)

    hook_cfg = dict(
        type='YOLOXLrUpdaterHook',
        warmup='exp',
        by_epoch=False,
        warmup_by_epoch=True,
        warmup_ratio=0.01,
        warmup_iters=5,  # 5 epoch
        last_epoch=15,
        min_lr_ratio=0.05)
    runner.register_hook_from_cfg(hook_cfg)
    runner.register_hook_from_cfg(dict(type='IterTimerHook'))
    runner.register_hook(IterTimerHook())

    # add pavi hook
    hook = PaviLoggerHook(interval=1, add_graph=False, add_last_ckpt=True)
    runner.register_hook(hook)
    runner.run([loader], [('train', 1)])
    shutil.rmtree(runner.work_dir)

    # TODO: use a more elegant way to check values
    assert hasattr(hook, 'writer')
    if multi_optimziers:
        calls = [
            call(
                'train', {
                    'learning_rate/model1': 0.0,
                    'learning_rate/model2': 0.0,
                    'momentum/model1': 0.95,
                    'momentum/model2': 0.9
                }, 1),
            call(
                'train', {
                    'learning_rate/model1': 0.000144,
                    'learning_rate/model2': 0.000144,
                    'momentum/model1': 0.95,
                    'momentum/model2': 0.9
                }, 7),
            call(
                'train', {
                    'learning_rate/model1': 0.000324,
                    'learning_rate/model2': 0.000324,
                    'momentum/model1': 0.95,
                    'momentum/model2': 0.9
                }, 10)
        ]
    else:
        calls = [
            call('train', {
                'learning_rate': 0.0,
                'momentum': 0.95
            }, 1),
            call('train', {
                'learning_rate': 0.000144,
                'momentum': 0.95
            }, 7),
            call('train', {
                'learning_rate': 0.000324,
                'momentum': 0.95
            }, 10)
        ]
    hook.writer.add_scalars.assert_has_calls(calls, any_order=True)
