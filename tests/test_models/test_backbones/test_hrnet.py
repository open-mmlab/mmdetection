import pytest
import torch

from mmdet.models.backbones.hrnet import HRNet


def test_hourglass_backbone():
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
    feat = model(imgs)
    assert len(feat) == 4
    assert feat[0].shape == torch.Size([1, 32, 64, 64])
    assert feat[3].shape == torch.Size([1, 256, 8, 8])

    # Test single scale output
    model = HRNet(extra=extra, multiscale_output=False)
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 256, 256)
    feat = model(imgs)
    assert len(feat) == 1
    assert feat[0].shape == torch.Size([1, 32, 64, 64])
