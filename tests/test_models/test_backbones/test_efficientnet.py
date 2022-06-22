import pytest
import torch

from mmdet.models.backbones import EfficientNet


def test_efficientnet_backbone():
    """Test EfficientNet backbone."""
    with pytest.raises(AssertionError):
        # EfficientNet arch should be a key in EfficientNet.arch_settings
        EfficientNet(arch='c3')

    model = EfficientNet(arch='b0', out_indices=(0, 1, 2, 3, 4, 5, 6))
    model.train()

    imgs = torch.randn(2, 3, 32, 32)
    feat = model(imgs)
    assert len(feat) == 7
    assert feat[0].shape == torch.Size([2, 32, 16, 16])
    assert feat[1].shape == torch.Size([2, 16, 16, 16])
    assert feat[2].shape == torch.Size([2, 24, 8, 8])
    assert feat[3].shape == torch.Size([2, 40, 4, 4])
    assert feat[4].shape == torch.Size([2, 112, 2, 2])
    assert feat[5].shape == torch.Size([2, 320, 1, 1])
    assert feat[6].shape == torch.Size([2, 1280, 1, 1])
