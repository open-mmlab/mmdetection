# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
from torch.nn.modules.batchnorm import _BatchNorm

from mmdet.models.backbones.csp_darknet import CSPDarknet
from .utils import check_norm_state, is_norm


def test_csp_darknet_backbone():
    with pytest.raises(ValueError):
        # frozen_stages must in range(-1, len(arch_setting) + 1)
        CSPDarknet(frozen_stages=6)

    with pytest.raises(AssertionError):
        # out_indices in range(len(arch_setting) + 1)
        CSPDarknet(out_indices=[6])

    # Test CSPDarknet with first stage frozen
    frozen_stages = 1
    model = CSPDarknet(frozen_stages=frozen_stages)
    model.train()

    for mod in model.stem.modules():
        for param in mod.parameters():
            assert param.requires_grad is False
    for i in range(1, frozen_stages + 1):
        layer = getattr(model, f'stage{i}')
        for mod in layer.modules():
            if isinstance(mod, _BatchNorm):
                assert mod.training is False
        for param in layer.parameters():
            assert param.requires_grad is False

    # Test CSPDarknet with norm_eval=True
    model = CSPDarknet(norm_eval=True)
    model.train()

    assert check_norm_state(model.modules(), False)

    # Test CSPDarknet-P5 forward with widen_factor=0.5
    model = CSPDarknet(arch='P5', widen_factor=0.25, out_indices=range(0, 5))
    model.train()

    imgs = torch.randn(1, 3, 64, 64)
    feat = model(imgs)
    assert len(feat) == 5
    assert feat[0].shape == torch.Size((1, 16, 32, 32))
    assert feat[1].shape == torch.Size((1, 32, 16, 16))
    assert feat[2].shape == torch.Size((1, 64, 8, 8))
    assert feat[3].shape == torch.Size((1, 128, 4, 4))
    assert feat[4].shape == torch.Size((1, 256, 2, 2))

    # Test CSPDarknet-P6 forward with widen_factor=0.5
    model = CSPDarknet(
        arch='P6',
        widen_factor=0.25,
        out_indices=range(0, 6),
        spp_kernal_sizes=(3, 5, 7))
    model.train()

    imgs = torch.randn(1, 3, 128, 128)
    feat = model(imgs)
    assert feat[0].shape == torch.Size((1, 16, 64, 64))
    assert feat[1].shape == torch.Size((1, 32, 32, 32))
    assert feat[2].shape == torch.Size((1, 64, 16, 16))
    assert feat[3].shape == torch.Size((1, 128, 8, 8))
    assert feat[4].shape == torch.Size((1, 192, 4, 4))
    assert feat[5].shape == torch.Size((1, 256, 2, 2))

    # Test CSPDarknet forward with dict(type='ReLU')
    model = CSPDarknet(
        widen_factor=0.125, act_cfg=dict(type='ReLU'), out_indices=range(0, 5))
    model.train()

    imgs = torch.randn(1, 3, 64, 64)
    feat = model(imgs)
    assert len(feat) == 5
    assert feat[0].shape == torch.Size((1, 8, 32, 32))
    assert feat[1].shape == torch.Size((1, 16, 16, 16))
    assert feat[2].shape == torch.Size((1, 32, 8, 8))
    assert feat[3].shape == torch.Size((1, 64, 4, 4))
    assert feat[4].shape == torch.Size((1, 128, 2, 2))

    # Test CSPDarknet with BatchNorm forward
    model = CSPDarknet(widen_factor=0.125, out_indices=range(0, 5))
    for m in model.modules():
        if is_norm(m):
            assert isinstance(m, _BatchNorm)
    model.train()

    imgs = torch.randn(1, 3, 64, 64)
    feat = model(imgs)
    assert len(feat) == 5
    assert feat[0].shape == torch.Size((1, 8, 32, 32))
    assert feat[1].shape == torch.Size((1, 16, 16, 16))
    assert feat[2].shape == torch.Size((1, 32, 8, 8))
    assert feat[3].shape == torch.Size((1, 64, 4, 4))
    assert feat[4].shape == torch.Size((1, 128, 2, 2))

    # Test CSPDarknet with custom arch forward
    arch_ovewrite = [[32, 56, 3, True, False], [56, 224, 2, True, False],
                     [224, 512, 1, True, False]]
    model = CSPDarknet(
        arch_ovewrite=arch_ovewrite,
        widen_factor=0.25,
        out_indices=(0, 1, 2, 3))
    model.train()

    imgs = torch.randn(1, 3, 32, 32)
    feat = model(imgs)
    assert len(feat) == 4
    assert feat[0].shape == torch.Size((1, 8, 16, 16))
    assert feat[1].shape == torch.Size((1, 14, 8, 8))
    assert feat[2].shape == torch.Size((1, 56, 4, 4))
    assert feat[3].shape == torch.Size((1, 128, 2, 2))
