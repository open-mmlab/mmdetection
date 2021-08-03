"""Tests the hooks with runners.
CommandLine:
    pytest tests/test_utils/test_hook_eam.py
    xdoctest tests/test_utils/test_hook_eam.py zero
"""
import tempfile
import logging
import shutil
import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import constant_
from torch.utils.data import DataLoader

from mmcv.runner import (CheckpointHook, build_runner)

from mmdet.core.hook import ExpDecayEMAHook, LinerDecayEMAHook

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
    ema_hook = ExpDecayEMAHook(decay=0.9998, total_iter=1,
        skip_bn_running_stats=True, interval=2, resume_from=None)
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
    resume_ema_hook = ExpDecayEMAHook(decay=0.5, total_iter=10,
        skip_bn_running_stats=True, interval=1, resume_from=f'{work_dir}/epoch_1.pth')
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
                np.allclose(value.data.cpu().numpy().reshape(-1), desired_output, 1e-4)
    assert num_eam_params == 4
    shutil.rmtree(runner.work_dir)
    shutil.rmtree(work_dir)
