# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
from mmcv.utils import ConfigDict

from mmdet.models.utils.transformer import (DetrTransformerDecoder,
                                            DetrTransformerEncoder, PatchEmbed,
                                            Transformer)


def test_patchembed():
    B = 2
    H = 3
    W = 4
    C = 3
    embed_dims = 10
    kernel_size = 3
    stride = 1
    dummy_input = torch.rand(B, C, H, W)
    patch_merge_1 = PatchEmbed(
        in_channels=C,
        embed_dims=embed_dims,
        conv_type=None,
        kernel_size=kernel_size,
        stride=stride,
        padding=0,
        dilation=1,
        norm_cfg=None,
    )

    x1 = patch_merge_1(dummy_input)
    assert x1.shape == (2, 2, 10)

    B = 2
    H = 10
    W = 10
    C = 3
    embed_dims = 10
    kernel_size = 5
    stride = 2
    dummy_input = torch.rand(B, C, H, W)
    # test dilation
    patch_merge_2 = PatchEmbed(
        in_channels=C,
        embed_dims=embed_dims,
        conv_type=None,
        kernel_size=kernel_size,
        stride=stride,
        padding=0,
        dilation=2,
        norm_cfg=None,
    )

    patch_merge_2(dummy_input)

    stride = 2
    dummy_input = torch.rand(B, C, H, W)
    # test stride and norm
    patch_merge_2 = PatchEmbed(
        in_channels=C,
        embed_dims=embed_dims,
        conv_type=None,
        kernel_size=kernel_size,
        stride=stride,
        padding=0,
        dilation=2,
        norm_cfg=dict(type='LN'))

    patch_merge_2(dummy_input)


def test_detr_transformer_dencoder_encoder_layer():
    config = ConfigDict(
        dict(
            return_intermediate=True,
            num_layers=6,
            transformerlayers=dict(
                type='DetrTransformerDecoderLayer',
                attn_cfgs=dict(
                    type='MultiheadAttention',
                    embed_dims=256,
                    num_heads=8,
                    dropout=0.1),
                feedforward_channels=2048,
                ffn_dropout=0.1,
                operation_order=(
                    'norm',
                    'self_attn',
                    'norm',
                    'cross_attn',
                    'norm',
                    'ffn',
                ))))
    assert DetrTransformerDecoder(**config).layers[0].pre_norm
    assert len(DetrTransformerDecoder(**config).layers) == 6

    DetrTransformerDecoder(**config)
    with pytest.raises(AssertionError):
        config = ConfigDict(
            dict(
                return_intermediate=True,
                num_layers=6,
                transformerlayers=[
                    dict(
                        type='DetrTransformerDecoderLayer',
                        attn_cfgs=dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        feedforward_channels=2048,
                        ffn_dropout=0.1,
                        operation_order=('self_attn', 'norm', 'cross_attn',
                                         'norm', 'ffn', 'norm'))
                ] * 5))
        DetrTransformerDecoder(**config)

    config = ConfigDict(
        dict(
            num_layers=6,
            transformerlayers=dict(
                type='DetrTransformerDecoderLayer',
                attn_cfgs=dict(
                    type='MultiheadAttention',
                    embed_dims=256,
                    num_heads=8,
                    dropout=0.1),
                feedforward_channels=2048,
                ffn_dropout=0.1,
                operation_order=('norm', 'self_attn', 'norm', 'cross_attn',
                                 'norm', 'ffn', 'norm'))))

    with pytest.raises(AssertionError):
        # len(operation_order) == 6
        DetrTransformerEncoder(**config)


def test_transformer():
    config = ConfigDict(
        dict(
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=6,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1)
                    ],
                    feedforward_channels=2048,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
            decoder=dict(
                type='DetrTransformerDecoder',
                return_intermediate=True,
                num_layers=6,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=dict(
                        type='MultiheadAttention',
                        embed_dims=256,
                        num_heads=8,
                        dropout=0.1),
                    feedforward_channels=2048,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')),
            )))
    transformer = Transformer(**config)
    transformer.init_weights()
