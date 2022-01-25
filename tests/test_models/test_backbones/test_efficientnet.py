import pytest
import torch
from torch.nn.modules import GroupNorm
from torch.nn.modules.batchnorm import _BatchNorm

from mmdet.models.backbones import EfficientNet
from .utils import check_norm_state, is_block, is_norm


def test_efficientnet_backbone():
    """Test EfficientNet backbone."""
    with pytest.raises(KeyError):
        # In EfficientNet: 0 <= scale <= 5
        EfficientNet(scale=6)

    with pytest.raises(AssertionError):
        # In EfficientNet: 1 <= out_indices <= 6
        EfficientNet(scale=3, out_indices=(2, 4, 7))

    # Test EfficientNet norm_eval=True
    model = EfficientNet(scale=3, norm_eval=True)
    model.train()
    assert check_norm_state(model.modules(), False)

    # Test EfficientNet with torchvision pretrained weight
    model = EfficientNet(scale=3, norm_eval=True,
                         pretrained='/checkpoints/converted_b3_2.pyth')
    model.train()
    assert check_norm_state(model.modules(), False)

    # Test EfficientNet with first stage frozen
    frozen_stages = 1
    model = EfficientNet(scale=3, frozen_stages=frozen_stages)
    model.init_weights()
    model.train()
    for layer in [model.conv1]:
        for param in layer.parameters():
            assert param.requires_grad is False
    for i in range(1, frozen_stages + 1):
        layer = getattr(model, f'layer{i}')
        for mod in layer.modules():
            if isinstance(mod, _BatchNorm):
                assert mod.training is False
        for param in layer.parameters():
            assert param.requires_grad is False

    # Test EfficientNet0 forward
    model = EfficientNet(scale=0, stem_channels=40)
    model.train()

    inputs = torch.rand(1, 3, 32, 32)
    feat = model.forward(inputs)
    assert len(feat) == 3
    assert feat[0].shape == torch.Size([1, 40, 4, 4])
    assert feat[1].shape == torch.Size([1, 112, 2, 2])
    assert feat[2].shape == torch.Size([1, 320, 1, 1])

    # Test EfficientNet0 with checkpoint forward
    model = EfficientNet(scale=0, with_cp=True)
    for m in model.modules():
        if is_block(m):
            assert m.with_cp

    # Test EfficientNet with BatchNorm forward
    model = EfficientNet(scale=3)
    for m in model.modules():
        if is_norm(m):
            assert isinstance(m, _BatchNorm)
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 32, 32)
    feat = model.forward(imgs)
    assert len(feat) == 3
    assert feat[0].shape == torch.Size([1, 48, 4, 4])
    assert feat[1].shape == torch.Size([1, 136, 2, 2])
    assert feat[2].shape == torch.Size([1, 384, 1, 1])

    # Test EfficientNet with checkpoint forward
    model = EfficientNet(scale=3, with_cp=True)
    for m in model.modules():
        if is_block(m):
            assert m.with_cp
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 32, 32)
    feat = model.forward(imgs)
    assert len(feat) == 3
    assert feat[0].shape == torch.Size([1, 48, 4, 4])
    assert feat[1].shape == torch.Size([1, 136, 2, 2])
    assert feat[2].shape == torch.Size([1, 384, 1, 1])

    # Test EfficientNet with GroupNorm forward
    model = EfficientNet(
        scale=3, norm_cfg=dict(type='GN', num_groups=2, requires_grad=True))
    for m in model.modules():
        if is_norm(m):
            assert isinstance(m, GroupNorm)
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 32, 32)
    feat = model.forward(imgs)
    assert len(feat) == 3
    assert feat[0].shape == torch.Size([1, 48, 4, 4])
    assert feat[1].shape == torch.Size([1, 136, 2, 2])
    assert feat[2].shape == torch.Size([1, 384, 1, 1])
