import logging
import shutil
import sys
import tempfile
from unittest.mock import MagicMock, call

import numpy as np
import pytest
import torch
import torch.nn as nn
from mmcv.runner import (CheckpointHook, IterTimerHook, PaviLoggerHook,
                         build_runner)
from torch.nn.init import constant_
from torch.utils.data import DataLoader

from mmdet.core.hook import ExpMomentumEMAHook, YOLOXLrUpdaterHook
from mmdet.core.hook.sync_norm_hook import SyncNormHook
from mmdet.core.hook.sync_random_size_hook import SyncRandomSizeHook


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
    YOLOXLrUpdaterHook(0, min_lr_ratio=0.05)

    sys.modules['pavi'] = MagicMock()
    loader = DataLoader(torch.ones((10, 2)))
    runner = _build_demo_runner(multi_optimziers=multi_optimziers)

    hook_cfg = dict(
        type='YOLOXLrUpdaterHook',
        warmup='exp',
        by_epoch=False,
        warmup_by_epoch=True,
        warmup_ratio=1,
        warmup_iters=5,  # 5 epoch
        num_last_epochs=15,
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
                    'learning_rate/model1': 8.000000000000001e-06,
                    'learning_rate/model2': 4.000000000000001e-06,
                    'momentum/model1': 0.95,
                    'momentum/model2': 0.9
                }, 1),
            call(
                'train', {
                    'learning_rate/model1': 0.00039200000000000004,
                    'learning_rate/model2': 0.00019600000000000002,
                    'momentum/model1': 0.95,
                    'momentum/model2': 0.9
                }, 7),
            call(
                'train', {
                    'learning_rate/model1': 0.0008000000000000001,
                    'learning_rate/model2': 0.0004000000000000001,
                    'momentum/model1': 0.95,
                    'momentum/model2': 0.9
                }, 10)
        ]
    else:
        calls = [
            call('train', {
                'learning_rate': 8.000000000000001e-06,
                'momentum': 0.95
            }, 1),
            call('train', {
                'learning_rate': 0.00039200000000000004,
                'momentum': 0.95
            }, 7),
            call('train', {
                'learning_rate': 0.0008000000000000001,
                'momentum': 0.95
            }, 10)
        ]
    hook.writer.add_scalars.assert_has_calls(calls, any_order=True)


def test_ema_hook():
    """xdoctest -m tests/test_hooks.py test_ema_hook."""

    class DemoModel(nn.Module):

        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(
                in_channels=1,
                out_channels=2,
                kernel_size=1,
                padding=1,
                bias=True)
            self.bn = nn.BatchNorm2d(2)

            self._init_weight()

        def _init_weight(self):
            constant_(self.conv.weight, 0)
            constant_(self.conv.bias, 0)
            constant_(self.bn.weight, 0)
            constant_(self.bn.bias, 0)

        def forward(self, x):
            return self.bn(self.conv(x)).sum()

        def train_step(self, x, optimizer, **kwargs):
            return dict(loss=self(x))

        def val_step(self, x, optimizer, **kwargs):
            return dict(loss=self(x))

    loader = DataLoader(torch.ones((1, 1, 1, 1)))
    runner = _build_demo_runner()
    demo_model = DemoModel()
    runner.model = demo_model
    ema_hook = ExpMomentumEMAHook(
        momentum=0.0002,
        total_iter=1,
        skip_buffers=True,
        interval=2,
        resume_from=None)
    checkpointhook = CheckpointHook(interval=1, by_epoch=True)
    runner.register_hook(ema_hook, priority='HIGHEST')
    runner.register_hook(checkpointhook)
    runner.run([loader, loader], [('train', 1), ('val', 1)])
    checkpoint = torch.load(f'{runner.work_dir}/epoch_1.pth')
    num_eam_params = 0
    for name, value in checkpoint['state_dict'].items():
        if 'ema' in name:
            num_eam_params += 1
            value.fill_(1)
    assert num_eam_params == 4
    torch.save(checkpoint, f'{runner.work_dir}/epoch_1.pth')

    work_dir = runner.work_dir
    resume_ema_hook = ExpMomentumEMAHook(
        momentum=0.5,
        total_iter=10,
        skip_buffers=True,
        interval=1,
        resume_from=f'{work_dir}/epoch_1.pth')
    runner = _build_demo_runner(max_epochs=2)
    runner.model = demo_model
    runner.register_hook(resume_ema_hook, priority='HIGHEST')
    checkpointhook = CheckpointHook(interval=1, by_epoch=True)
    runner.register_hook(checkpointhook)
    runner.run([loader, loader], [('train', 1), ('val', 1)])
    checkpoint = torch.load(f'{runner.work_dir}/epoch_2.pth')
    num_eam_params = 0
    desired_output = [0.9094, 0.9094]
    for name, value in checkpoint['state_dict'].items():
        if 'ema' in name:
            num_eam_params += 1
            assert value.sum() == 2
        else:
            if ('weight' in name) or ('bias' in name):
                np.allclose(value.data.cpu().numpy().reshape(-1),
                            desired_output, 1e-4)
    assert num_eam_params == 4
    shutil.rmtree(runner.work_dir)
    shutil.rmtree(work_dir)


def test_sync_norm_hook():
    # Only used to prevent program errors
    SyncNormHook()

    loader = DataLoader(torch.ones((5, 2)))
    runner = _build_demo_runner()
    runner.register_hook_from_cfg(dict(type='SyncNormHook'))
    runner.run([loader, loader], [('train', 1), ('val', 1)])
    shutil.rmtree(runner.work_dir)


def test_sync_random_size_hook():
    # Only used to prevent program errors
    SyncRandomSizeHook()

    loader = DataLoader(torch.ones((5, 2)))
    runner = _build_demo_runner()
    runner.register_hook_from_cfg(dict(type='SyncRandomSizeHook'))
    runner.run([loader, loader], [('train', 1), ('val', 1)])
    shutil.rmtree(runner.work_dir)


@pytest.mark.parametrize('set_loss', [
    dict(set_loss_nan=False, set_loss_inf=False),
    dict(set_loss_nan=True, set_loss_inf=False),
    dict(set_loss_nan=False, set_loss_inf=True)
])
def test_check_invalid_loss_hook(set_loss):
    # Check whether loss is valid during training.

    class DemoModel(nn.Module):

        def __init__(self, set_loss_nan=False, set_loss_inf=False):
            super().__init__()
            self.set_loss_nan = set_loss_nan
            self.set_loss_inf = set_loss_inf
            self.linear = nn.Linear(2, 1)

        def forward(self, x):
            return self.linear(x)

        def train_step(self, x, optimizer, **kwargs):
            if self.set_loss_nan:
                return dict(loss=torch.tensor(float('nan')))
            elif self.set_loss_inf:
                return dict(loss=torch.tensor(float('inf')))
            else:
                return dict(loss=self(x))

    loader = DataLoader(torch.ones((5, 2)))
    runner = _build_demo_runner()

    demo_model = DemoModel(**set_loss)
    runner.model = demo_model
    runner.register_hook_from_cfg(
        dict(type='CheckInvalidLossHook', interval=1))
    if not set_loss['set_loss_nan'] \
            and not set_loss['set_loss_inf']:
        # check loss is valid
        runner.run([loader], [('train', 1)])
    else:
        # check loss is nan or inf
        with pytest.raises(AssertionError):
            runner.run([loader], [('train', 1)])
    shutil.rmtree(runner.work_dir)
