# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
import torch.nn.functional as F
from mmengine.model import constant_init

from mmdet.models.layers import DyReLU, SELayer


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


def test_dyrelu():
    with pytest.raises(AssertionError):
        # act_cfg sequence length must equal to 2
        DyReLU(channels=32, act_cfg=(dict(type='ReLU'), ))

    with pytest.raises(AssertionError):
        # act_cfg sequence must be a tuple of dict
        DyReLU(channels=32, act_cfg=[dict(type='ReLU'), dict(type='ReLU')])

    # Test DyReLU forward
    layer = DyReLU(channels=32)
    layer.init_weights()
    layer.train()
    x = torch.randn((1, 32, 10, 10))
    x_out = layer(x)
    assert x_out.shape == torch.Size((1, 32, 10, 10))

    # DyReLU should act as standard (static) ReLU
    # when eliminating the effect of SE-like module
    layer = DyReLU(channels=32)
    constant_init(layer.conv2.conv, 0)
    layer.train()
    x = torch.randn((1, 32, 10, 10))
    x_out = layer(x)
    relu_out = F.relu(x)
    assert torch.equal(x_out, relu_out)
