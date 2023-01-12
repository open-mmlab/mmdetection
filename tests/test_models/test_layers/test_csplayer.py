import pytest
import torch

from mmdet.models.layers.csp_layer import *


def test_c2flayer():
    # Test SELayer forward
    layer = C2fLayer(3, 64, 1)
    layer.init_weights()
    layer.train()

    x = torch.randn(1, 3, 640, 640)
    x_out = layer(x)
    assert x_out.shape == torch.Size((1, 64, 640, 640))
