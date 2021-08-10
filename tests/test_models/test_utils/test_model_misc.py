import numpy as np
import torch

from mmdet.models.utils import interpolate_as


def test_interpolate_as():
    logits = torch.rand((1, 5, 4, 4))
    targets = torch.rand((1, 1, 16, 16))

    # Test 4D logits and targets
    result = interpolate_as(logits, targets)
    assert result.shape == torch.Size((1, 5, 16, 16))

    # Test 3D targets
    result = interpolate_as(logits, targets.squeeze(0))
    assert result.shape == torch.Size((1, 5, 16, 16))

    # Test 3D logits
    result = interpolate_as(logits.squeeze(0), targets)
    assert result.shape == torch.Size((5, 16, 16))

    # Test type(target) == np.ndarray
    targets = np.random.rand(16, 16)
    result = interpolate_as(logits.squeeze(0), targets)
    assert result.shape == torch.Size((5, 16, 16))
