"""
CommandLine:
    pytest tests/test_center_pool.py
"""
import pytest
import torch

from mmdet.ops import CenterPool


def test_center_pool_device_and_dtypes_cpu():
    """
    CommandLine:
        xdoctest -m tests/test_center_pool.py \
            test_center_pool_device_and_dtypes_cpu
    """
    with pytest.raises(AssertionError):
        # pool mode must in ['bottom', 'left', 'right', 'top']
        pool = CenterPool('center')

    lr_tensor = torch.tensor([[[[0, 0, 0, 0, 0], [2, 1, 3, 0, 2],
                                [5, 4, 1, 1, 6], [0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0]]]])
    tb_tensor = torch.tensor([[[[0, 3, 1, 0, 0], [0, 1, 1, 0, 0],
                                [0, 3, 4, 0, 0], [0, 2, 2, 0, 0],
                                [0, 0, 2, 0, 0]]]])
    # Left Pool
    left_answer = torch.tensor([[[[0, 0, 0, 0, 0], [3, 3, 3, 2, 2],
                                  [6, 6, 6, 6, 6], [0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0]]]])
    pool = CenterPool('left')
    left_tensor = pool(lr_tensor)
    assert left_tensor.type() == lr_tensor.type()
    assert torch.equal(left_tensor, left_answer)
    # Right Pool
    right_answer = torch.tensor([[[[0, 0, 0, 0, 0], [2, 2, 3, 3, 3],
                                   [5, 5, 5, 5, 6], [0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0]]]])
    pool = CenterPool('right')
    right_tensor = pool(lr_tensor)
    assert right_tensor.type() == lr_tensor.type()
    assert torch.equal(right_tensor, right_answer)
    # Top Pool
    top_answer = torch.tensor([[[[0, 3, 4, 0, 0], [0, 3, 4, 0, 0],
                                 [0, 3, 4, 0, 0], [0, 2, 2, 0, 0],
                                 [0, 0, 2, 0, 0]]]])
    pool = CenterPool('top')
    top_tensor = pool(tb_tensor)
    assert top_tensor.type() == tb_tensor.type()
    assert torch.equal(top_tensor, top_answer)
    # Bottom Pool
    bottom_answer = torch.tensor([[[[0, 3, 1, 0, 0], [0, 3, 1, 0, 0],
                                    [0, 3, 4, 0, 0], [0, 3, 4, 0, 0],
                                    [0, 3, 4, 0, 0]]]])
    pool = CenterPool('bottom')
    bottom_tensor = pool(tb_tensor)
    assert bottom_tensor.type() == tb_tensor.type()
    assert torch.equal(bottom_tensor, bottom_answer)
