# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from torch.autograd import gradcheck

from mmdet.models.utils import interpolate_as, sigmoid_geometric_mean


def test_interpolate_as():
    source = torch.rand((1, 5, 4, 4))
    target = torch.rand((1, 1, 16, 16))

    # Test 4D source and target
    result = interpolate_as(source, target)
    assert result.shape == torch.Size((1, 5, 16, 16))

    # Test 3D target
    result = interpolate_as(source, target.squeeze(0))
    assert result.shape == torch.Size((1, 5, 16, 16))

    # Test 3D source
    result = interpolate_as(source.squeeze(0), target)
    assert result.shape == torch.Size((5, 16, 16))

    # Test type(target) == np.ndarray
    target = np.random.rand(16, 16)
    result = interpolate_as(source.squeeze(0), target)
    assert result.shape == torch.Size((5, 16, 16))


def test_sigmoid_geometric_mean():
    x = torch.randn(20, 20, dtype=torch.double, requires_grad=True)
    y = torch.randn(20, 20, dtype=torch.double, requires_grad=True)
    inputs = (x, y)
    test = gradcheck(sigmoid_geometric_mean, inputs, eps=1e-6, atol=1e-4)
    assert test
