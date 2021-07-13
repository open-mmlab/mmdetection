import pytest
import torch

from mmdet.models.plugins import DropBlock


def test_dropblock():
    feat = torch.rand(1, 1, 11, 11)
    drop_prob = 1.0
    dropblock = DropBlock(drop_prob, block_size=11, warmup_iters=0)
    out_feat = dropblock(feat)
    assert (out_feat == 0).all() and out_feat.shape == feat.shape
    drop_prob = 0.5
    dropblock = DropBlock(drop_prob, block_size=5, warmup_iters=0)
    out_feat = dropblock(feat)
    assert out_feat.shape == feat.shape

    # drop_prob must be (0,1]
    with pytest.raises(AssertionError):
        DropBlock(1.5, 3)

    # block_size cannot be an even number
    with pytest.raises(AssertionError):
        DropBlock(0.5, 2)

    # warmup_iters cannot be less than 0
    with pytest.raises(AssertionError):
        DropBlock(0.5, 3, -1)
