import pytest
import torch

from mmdet.models.utils import SELayer


def test_se_layer():
    with pytest.raises(AssertionError):
        # act_cfg sequence length must equal to 2
        SELayer(channels=32, act_cfg=(dict(type='ReLU'), ))

    with pytest.raises(AssertionError):
        # act_cfg sequence must be a tuple of dict
        SELayer(channels=32, act_cfg=[dict(type='ReLU'), dict(type='ReLU')])

    # Test SELayer forward
    layer = SELayer(channels=32)
    layer.init_weights()
    layer.train()

    x = torch.randn((1, 32, 10, 10))
    x_out = layer(x)
    assert x_out.shape == torch.Size((1, 32, 10, 10))
