# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
from mmcv import ConfigDict
from mmcv.cnn import build_plugin_layer

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


def test_pixeldecoder():
    base_channels = 64
    pixel_decoder_cfg = ConfigDict(
        dict(
            type='PixelDecoder',
            in_channels=[base_channels * 2**i for i in range(4)],
            feat_channels=base_channels,
            out_channels=base_channels,
            norm_cfg=dict(type='GN', num_groups=32),
            act_cfg=dict(type='ReLU')))
    self = build_plugin_layer(pixel_decoder_cfg)[1]
    img_metas = [{}, {}]
    feats = [
        torch.rand((2, base_channels * 2**i, 4 * 2**(3 - i), 5 * 2**(3 - i)))
        for i in range(4)
    ]
    mask_feature, memory = self(feats, img_metas)

    assert (memory == feats[-1]).all()
    assert mask_feature.shape == feats[0].shape


def test_transformerencoderpixeldecoer():
    base_channels = 64
    pixel_decoder_cfg = ConfigDict(
        dict(
            type='TransformerEncoderPixelDecoder',
            in_channels=[base_channels * 2**i for i in range(4)],
            feat_channels=base_channels,
            out_channels=base_channels,
            norm_cfg=dict(type='GN', num_groups=32),
            act_cfg=dict(type='ReLU'),
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=6,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='MultiheadAttention',
                        embed_dims=base_channels,
                        num_heads=8,
                        attn_drop=0.1,
                        proj_drop=0.1,
                        dropout_layer=None,
                        batch_first=False),
                    ffn_cfgs=dict(
                        embed_dims=base_channels,
                        feedforward_channels=base_channels * 8,
                        num_fcs=2,
                        act_cfg=dict(type='ReLU', inplace=True),
                        ffn_drop=0.1,
                        dropout_layer=None,
                        add_identity=True),
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'),
                    norm_cfg=dict(type='LN'),
                    init_cfg=None,
                    batch_first=False),
                init_cfg=None),
            positional_encoding=dict(
                type='SinePositionalEncoding',
                num_feats=base_channels // 2,
                normalize=True)))
    self = build_plugin_layer(pixel_decoder_cfg)[1]
    img_metas = [{
        'batch_input_shape': (128, 160),
        'img_shape': (120, 160, 3),
    }, {
        'batch_input_shape': (128, 160),
        'img_shape': (125, 160, 3),
    }]
    feats = [
        torch.rand((2, base_channels * 2**i, 4 * 2**(3 - i), 5 * 2**(3 - i)))
        for i in range(4)
    ]
    mask_feature, memory = self(feats, img_metas)

    assert memory.shape[-2:] == feats[-1].shape[-2:]
    assert mask_feature.shape == feats[0].shape
