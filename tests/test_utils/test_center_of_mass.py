import pytest
import torch

from mmdet.core.utils import center_of_mass


@pytest.mark.parametrize('mask', [
    torch.ones((28, 28)),
    torch.zeros((28, 28)),
    torch.rand(28, 28) > 0.5,
    torch.tensor([[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]])
])
def test_center_of_mass(mask):
    center_h, center_w = center_of_mass(mask)
    if mask.shape[0] == 4:
        assert center_h == 1.5
        assert center_w == 1.5
    assert isinstance(center_h, torch.Tensor) \
           and isinstance(center_w, torch.Tensor)
    assert 0 <= center_h <= 28 \
           and 0 <= center_w <= 28
