# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmdet.models.backbones import Res2Net
from mmdet.models.backbones.res2net import Bottle2neck
from .utils import is_block


def test_res2net_bottle2neck():
    with pytest.raises(AssertionError):
        # Style must be in ['pytorch', 'caffe']
        Bottle2neck(64, 64, base_width=26, scales=4, style='tensorflow')

    with pytest.raises(AssertionError):
        # Scale must be larger than 1
        Bottle2neck(64, 64, base_width=26, scales=1, style='pytorch')

    # Test Res2Net Bottle2neck structure
    block = Bottle2neck(
        64, 64, base_width=26, stride=2, scales=4, style='pytorch')
    assert block.scales == 4

    # Test Res2Net Bottle2neck with DCN
    dcn = dict(type='DCN', deform_groups=1, fallback_on_stride=False)
    with pytest.raises(AssertionError):
        # conv_cfg must be None if dcn is not None
        Bottle2neck(
            64,
            64,
            base_width=26,
            scales=4,
            dcn=dcn,
            conv_cfg=dict(type='Conv'))
    Bottle2neck(64, 64, dcn=dcn)

    # Test Res2Net Bottle2neck forward
    block = Bottle2neck(64, 16, base_width=26, scales=4)
    x = torch.randn(1, 64, 56, 56)
    x_out = block(x)
    assert x_out.shape == torch.Size([1, 64, 56, 56])


def test_res2net_backbone():
    with pytest.raises(KeyError):
        # Res2Net depth should be in [50, 101, 152]
        Res2Net(depth=18)

    # Test Res2Net with scales 4, base_width 26
    model = Res2Net(depth=50, scales=4, base_width=26)
    for m in model.modules():
        if is_block(m):
            assert m.scales == 4
    model.train()

    imgs = torch.randn(1, 3, 32, 32)
    feat = model(imgs)
    assert len(feat) == 4
    assert feat[0].shape == torch.Size([1, 256, 8, 8])
    assert feat[1].shape == torch.Size([1, 512, 4, 4])
    assert feat[2].shape == torch.Size([1, 1024, 2, 2])
    assert feat[3].shape == torch.Size([1, 2048, 1, 1])
