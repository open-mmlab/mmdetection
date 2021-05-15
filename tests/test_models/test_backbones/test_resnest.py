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
        64, 256, radix=2, reduction_factor=4, stride=2, style='pytorch')
    assert block.avd_layer.stride == 2
    assert block.conv2.channels == 256

    # Test ResNeSt Bottleneck forward
    block = BottleneckS(64, 16, radix=2, reduction_factor=4)
    x = torch.randn(2, 64, 56, 56)
    x_out = block(x)
    assert x_out.shape == torch.Size([2, 64, 56, 56])


def test_resnest_backbone():
    with pytest.raises(KeyError):
        # ResNeSt depth should be in [50, 101, 152, 200]
        ResNeSt(depth=18)

    # Test ResNeSt with radix 2, reduction_factor 4
    model = ResNeSt(
        depth=50, radix=2, reduction_factor=4, out_indices=(0, 1, 2, 3))
    model.init_weights()
    model.train()

    imgs = torch.randn(2, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 4
    assert feat[0].shape == torch.Size([2, 256, 56, 56])
    assert feat[1].shape == torch.Size([2, 512, 28, 28])
    assert feat[2].shape == torch.Size([2, 1024, 14, 14])
    assert feat[3].shape == torch.Size([2, 2048, 7, 7])
