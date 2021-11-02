# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmdet.models.backbones import ResNeSt
from mmdet.models.backbones.resnest import Bottleneck as BottleneckS


def test_resnest_bottleneck():
    with pytest.raises(AssertionError):
        # Style must be in ['pytorch', 'caffe']
        BottleneckS(64, 64, radix=2, reduction_factor=4, style='tensorflow')

    # Test ResNeSt Bottleneck structure
    block = BottleneckS(
        2, 4, radix=2, reduction_factor=4, stride=2, style='pytorch')
    assert block.avd_layer.stride == 2
    assert block.conv2.channels == 4

    # Test ResNeSt Bottleneck forward
    block = BottleneckS(16, 4, radix=2, reduction_factor=4)
    x = torch.randn(2, 16, 56, 56)
    x_out = block(x)
    assert x_out.shape == torch.Size([2, 16, 56, 56])


def test_resnest_backbone():
    with pytest.raises(KeyError):
        # ResNeSt depth should be in [50, 101, 152, 200]
        ResNeSt(depth=18)

    # Test ResNeSt with radix 2, reduction_factor 4
    model = ResNeSt(
        depth=50,
        base_channels=4,
        radix=2,
        reduction_factor=4,
        out_indices=(0, 1, 2, 3))
    model.train()

    imgs = torch.randn(2, 3, 32, 32)
    feat = model(imgs)
    assert len(feat) == 4
    assert feat[0].shape == torch.Size([2, 16, 8, 8])
    assert feat[1].shape == torch.Size([2, 32, 4, 4])
    assert feat[2].shape == torch.Size([2, 64, 2, 2])
    assert feat[3].shape == torch.Size([2, 128, 1, 1])
