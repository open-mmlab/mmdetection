import pytest
import torch

from mmdet.models.backbones.hourglass import HourglassNet


def test_hourglass_backbone():
    with pytest.raises(AssertionError):
        # HourglassNet's num_stacks should larger than 0
        HourglassNet(num_stacks=0)

    with pytest.raises(AssertionError):
        # len(stage_channels) should equal len(stage_blocks)
        HourglassNet(
            stage_channels=[256, 256, 384, 384, 384],
            stage_blocks=[2, 2, 2, 2, 2, 4])

    with pytest.raises(AssertionError):
        # len(stage_channels) should lagrer than downsample_times
        HourglassNet(
            downsample_times=5,
            stage_channels=[256, 256, 384, 384, 384],
            stage_blocks=[2, 2, 2, 2, 2])

    # Test HourglassNet-52
    model = HourglassNet(num_stacks=1)
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 256, 256)
    feat = model(imgs)
    assert len(feat) == 1
    assert feat[0].shape == torch.Size([1, 256, 64, 64])

    # Test HourglassNet-104
    model = HourglassNet(num_stacks=2)
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 256, 256)
    feat = model(imgs)
    assert len(feat) == 2
    assert feat[0].shape == torch.Size([1, 256, 64, 64])
    assert feat[1].shape == torch.Size([1, 256, 64, 64])
