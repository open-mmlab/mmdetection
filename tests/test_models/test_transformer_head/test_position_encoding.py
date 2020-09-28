import pytest
import torch

from mmdet.models.transformer_head.position_encoding import (
    PositionEmbeddingLearned, PositionEmbeddingSine)


def test_position_encoding_sine(num_pos_feats=16, batch_size=2):
    with pytest.raises(ValueError):
        module = PositionEmbeddingSine(num_pos_feats, scale=3.)

    module = PositionEmbeddingSine(num_pos_feats)
    h, w = 10, 6
    mask = torch.rand(batch_size, h, w) > 0.5
    assert not module.normalize
    out = module(mask)
    assert out.shape == (batch_size, num_pos_feats * 2, h, w)

    # set normalize
    module = PositionEmbeddingSine(num_pos_feats, normalize=True)
    assert module.normalize
    out = module(mask)
    assert out.shape == (batch_size, num_pos_feats * 2, h, w)


def test_position_encoding_learned(num_pos_feats=16,
                                   num_embed=10,
                                   batch_size=2):
    module = PositionEmbeddingLearned(num_pos_feats, num_embed)
    assert module.row_embed.weight.shape == (num_embed, num_pos_feats)
    assert module.col_embed.weight.shape == (num_embed, num_pos_feats)
    h, w = 10, 6
    mask = torch.rand(batch_size, h, w) > 0.5
    out = module(mask)
    assert out.shape == (batch_size, num_pos_feats * 2, h, w)
