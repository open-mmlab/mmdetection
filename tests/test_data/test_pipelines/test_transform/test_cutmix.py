# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
from mmcv.utils import build_from_cfg
from mmdet.datasets.builder import PIPELINES

pipelines_cfgs = [
    dict(type='CutMix', alpha=1., prob=1.),
]


def test_cutmix():
    imgs = torch.randn(4, 3, 32, 32)
    labels = torch.randint(0, 10, (4, ))

    # Test cutmix
    pipelines_cfg = dict(type='CutMix', alpha=1., num_classes=10, prob=1.)
    cutmix_module = build_from_cfg(pipelines_cfg, PIPELINES)
    mixed_imgs, mixed_labels = cutmix_module(imgs, labels)
    assert mixed_imgs.shape == torch.Size((4, 3, 32, 32))
    assert mixed_labels.shape == torch.Size((4, 10))


@pytest.mark.parametrize('cfg', pipelines_cfgs)
def test_binary_augment(cfg):

    cfg_ = dict(num_classes=1, **cfg)
    cutmix_module = build_from_cfg(cfg_, PIPELINES)

    imgs = torch.randn(4, 3, 32, 32)
    labels = torch.randint(0, 2, (4, 1)).float()

    mixed_imgs, mixed_labels = cutmix_module(imgs, labels)
    assert mixed_imgs.shape == torch.Size((4, 3, 32, 32))
    assert mixed_labels.shape == torch.Size((4, 1))


@pytest.mark.parametrize('cfg', pipelines_cfgs)
def test_multilabel_augment(cfg):

    cfg_ = dict(num_classes=10, **cfg)
    cutmix_module = build_from_cfg(cfg_, PIPELINES)

    imgs = torch.randn(4, 3, 32, 32)
    labels = torch.randint(0, 2, (4, 10)).float()

    mixed_imgs, mixed_labels = cutmix_module(imgs, labels)
    assert mixed_imgs.shape == torch.Size((4, 3, 32, 32))
    assert mixed_labels.shape == torch.Size((4, 10))
