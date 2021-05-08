import pytest
import torch

from mmdet.models.backbones import DLANet
from .utils import check_norm_state


def test_dlanet_backbone():
    """Test resnet backbone."""
    with pytest.raises(KeyError):
        # ResNet depth should be in [34,]
        DLANet(20)

    with pytest.raises(TypeError):
        # pretrained must be a string path
        model = DLANet(34, pretrained=0)
        model.init_weights()

    # Test DLANet34 with pretrained weight
    model = DLANet(
        depth=34,
        pretrained='http://dl.yf.io/dla/models/imagenet/dla34-ba72cf86.pth')
    model.init_weights()
    model.train()
    assert check_norm_state(model.modules(), True)

    # Test DLANet34 forward
    model = DLANet(34)
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 6
    assert feat[0].shape == torch.Size([1, 16, 224, 224])
    assert feat[1].shape == torch.Size([1, 32, 112, 112])
    assert feat[2].shape == torch.Size([1, 64, 56, 56])
    assert feat[3].shape == torch.Size([1, 128, 28, 28])
    assert feat[4].shape == torch.Size([1, 256, 14, 14])
    assert feat[5].shape == torch.Size([1, 512, 7, 7])
