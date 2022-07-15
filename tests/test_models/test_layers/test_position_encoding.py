# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmdet.models.layers import (LearnedPositionalEncoding,
                                 SinePositionalEncoding)


def test_sine_positional_encoding(num_feats=16, batch_size=2):
    # test invalid type of scale
    with pytest.raises(AssertionError):
        module = SinePositionalEncoding(
            num_feats, scale=(3., ), normalize=True)

    module = SinePositionalEncoding(num_feats)
    h, w = 10, 6
    mask = (torch.rand(batch_size, h, w) > 0.5).to(torch.int)
    assert not module.normalize
    out = module(mask)
    assert out.shape == (batch_size, num_feats * 2, h, w)

    # set normalize
    module = SinePositionalEncoding(num_feats, normalize=True)
    assert module.normalize
    out = module(mask)
    assert out.shape == (batch_size, num_feats * 2, h, w)


def test_learned_positional_encoding(num_feats=16,
                                     row_num_embed=10,
                                     col_num_embed=10,
                                     batch_size=2):
    module = LearnedPositionalEncoding(num_feats, row_num_embed, col_num_embed)
    assert module.row_embed.weight.shape == (row_num_embed, num_feats)
    assert module.col_embed.weight.shape == (col_num_embed, num_feats)
    h, w = 10, 6
    mask = torch.rand(batch_size, h, w) > 0.5
    out = module(mask)
    assert out.shape == (batch_size, num_feats * 2, h, w)
