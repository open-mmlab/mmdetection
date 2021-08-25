# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmdet.models.backbones.hrnet import HRModule, HRNet
from mmdet.models.backbones.resnet import BasicBlock, Bottleneck


@pytest.mark.parametrize('block', [BasicBlock, Bottleneck])
def test_hrmodule(block):
    # Test multiscale forward
    num_channles = (32, 64)
    in_channels = [c * block.expansion for c in num_channles]
    hrmodule = HRModule(
        num_branches=2,
        blocks=block,
        in_channels=in_channels,
        num_blocks=(4, 4),
        num_channels=num_channles,
    )

    feats = [
        torch.randn(1, in_channels[0], 64, 64),
        torch.randn(1, in_channels[1], 32, 32)
    ]
    feats = hrmodule(feats)

    assert len(feats) == 2
    assert feats[0].shape == torch.Size([1, in_channels[0], 64, 64])
    assert feats[1].shape == torch.Size([1, in_channels[1], 32, 32])

    # Test single scale forward
    num_channles = (32, 64)
    in_channels = [c * block.expansion for c in num_channles]
    hrmodule = HRModule(
        num_branches=2,
        blocks=block,
        in_channels=in_channels,
        num_blocks=(4, 4),
        num_channels=num_channles,
        multiscale_output=False,
    )

    feats = [
        torch.randn(1, in_channels[0], 64, 64),
        torch.randn(1, in_channels[1], 32, 32)
    ]
    feats = hrmodule(feats)

    assert len(feats) == 1
    assert feats[0].shape == torch.Size([1, in_channels[0], 64, 64])


def test_hrnet_backbone():
    # only have 3 stages
    extra = dict(
        stage1=dict(
            num_modules=1,
            num_branches=1,
            block='BOTTLENECK',
            num_blocks=(4, ),
            num_channels=(64, )),
        stage2=dict(
            num_modules=1,
            num_branches=2,
            block='BASIC',
            num_blocks=(4, 4),
            num_channels=(32, 64)),
        stage3=dict(
            num_modules=4,
            num_branches=3,
            block='BASIC',
            num_blocks=(4, 4, 4),
            num_channels=(32, 64, 128)))

    with pytest.raises(AssertionError):
        # HRNet now only support 4 stages
        HRNet(extra=extra)
    extra['stage4'] = dict(
        num_modules=3,
        num_branches=3,  # should be 4
        block='BASIC',
        num_blocks=(4, 4, 4, 4),
        num_channels=(32, 64, 128, 256))

    with pytest.raises(AssertionError):
        # len(num_blocks) should equal num_branches
        HRNet(extra=extra)

    extra['stage4']['num_branches'] = 4

    # Test hrnetv2p_w32
    model = HRNet(extra=extra)
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 256, 256)
    feats = model(imgs)
    assert len(feats) == 4
    assert feats[0].shape == torch.Size([1, 32, 64, 64])
    assert feats[3].shape == torch.Size([1, 256, 8, 8])

    # Test single scale output
    model = HRNet(extra=extra, multiscale_output=False)
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 256, 256)
    feats = model(imgs)
    assert len(feats) == 1
    assert feats[0].shape == torch.Size([1, 32, 64, 64])
