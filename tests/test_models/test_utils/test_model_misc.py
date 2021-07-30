import numpy as np
import torch

from mmdet.models.utils import upsample_like


def test_upsample_like():
    logits = torch.rand((1, 5, 4, 4))
    targets = torch.rand((1, 1, 16, 16))

    # Test 4D logits and targets
    result = upsample_like(logits, targets)
    assert result.shape == torch.Size((1, 5, 16, 16))

    # Test 3D targets
    result = upsample_like(logits, targets.squeeze(0))
    assert result.shape == torch.Size((1, 5, 16, 16))

    # Test 3D logits
    result = upsample_like(logits.squeeze(0), targets)
    assert result.shape == torch.Size((5, 16, 16))

    # Test type(target) == np.ndarray
    targets = np.random.rand(16, 16)
    result = upsample_like(logits.squeeze(0), targets)
    assert result.shape == torch.Size((5, 16, 16))
