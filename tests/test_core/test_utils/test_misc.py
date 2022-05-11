import pytest
import torch

from mmdet.core import stack_batch


def test_stack_batch():
    # Input tensors must be a list
    with pytest.raises(AssertionError):
        stack_batch(torch.rand(1))

    # Input tensors must be a list and CHW 3D-tensor
    with pytest.raises(AssertionError):
        stack_batch([torch.rand((1, 2, 2, 2))])

    # The dimensions of the elements in the
    # input list must be the same
    with pytest.raises(AssertionError):
        stack_batch([torch.rand((1, 2, 2, 2)), torch.rand((3, 2, 2))])

    # The channel dimensions of the elements in the
    # input list must be the same
    with pytest.raises(AssertionError):
        stack_batch([torch.rand((2, 2, 2)), torch.rand((3, 2, 2))])

    out_tensor = stack_batch([torch.rand((2, 3, 2)), torch.rand((2, 2, 2))])
    assert out_tensor.shape == (2, 2, 3, 2)

    out_tensor = stack_batch([torch.rand(
        (2, 3, 2)), torch.rand((2, 2, 2))],
                             pad_size_divisor=5)
    assert out_tensor.shape == (2, 2, 5, 5)

    out_tensor = stack_batch([torch.rand(
        (2, 3, 2)), torch.rand((2, 2, 2))],
                             pad_size_divisor=5,
                             pad_value=1.2)
    assert out_tensor.shape == (2, 2, 5, 5)
