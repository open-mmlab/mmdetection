# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
from mmcv.utils import ConfigDict

from mmdet.models.utils.transformer import (DetrTransformerDecoder,
                                            DetrTransformerEncoder, PatchEmbed,
                                            PatchMerging, Transformer)


def test_patch_embed():
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
        pad_to_stride=False,
        conv_type=None,
        kernel_size=kernel_size,
        stride=stride,
        padding=0,
        dilation=1,
        norm_cfg=None,
    )

    x1, shape = patch_merge_1(dummy_input)
    # test out shape
    assert x1.shape == (2, 2, 10)
    # test outsize is correct
    assert shape == (1, 2)
    # test L = out_h * out_w
    assert shape[0] * shape[1] == x1.shape[1]

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
        pad_to_stride=False,
        conv_type=None,
        kernel_size=kernel_size,
        stride=stride,
        padding=0,
        dilation=2,
        norm_cfg=None,
    )

    x2, shape = patch_merge_2(dummy_input)
    # test out shape
    assert x2.shape == (2, 1, 10)
    # test outsize is correct
    assert shape == (1, 1)
    # test L = out_h * out_w
    assert shape[0] * shape[1] == x2.shape[1]

    stride = 2
    input_size = (10, 10)

    dummy_input = torch.rand(B, C, H, W)
    # test stride and norm
    patch_merge_3 = PatchEmbed(
        in_channels=C,
        embed_dims=embed_dims,
        pad_to_stride=False,
        conv_type=None,
        kernel_size=kernel_size,
        stride=stride,
        padding=0,
        dilation=2,
        norm_cfg=dict(type='LN'),
        input_size=input_size)

    x3, shape = patch_merge_3(dummy_input)
    # test out shape
    assert x3.shape == (2, 1, 10)
    # test outsize is correct
    assert shape == (1, 1)
    # test L = out_h * out_w
    assert shape[0] * shape[1] == x3.shape[1]

    # test thte init_out_size with nn.Unfold
    assert patch_merge_3.init_out_size[1] == (input_size[0] - 2 * 4 -
                                              1) // 2 + 1
    assert patch_merge_3.init_out_size[0] == (input_size[0] - 2 * 4 -
                                              1) // 2 + 1
    H = 11
    W = 12
    input_size = (H, W)
    dummy_input = torch.rand(B, C, H, W)
    # test stride and norm
    patch_merge_3 = PatchEmbed(
        in_channels=C,
        embed_dims=embed_dims,
        pad_to_stride=False,
        conv_type=None,
        kernel_size=kernel_size,
        stride=stride,
        padding=0,
        dilation=2,
        norm_cfg=dict(type='LN'),
        input_size=input_size)

    _, shape = patch_merge_3(dummy_input)
    # when input_size equal to real input
    # the out_size shoule be equal to `init_out_size`
    assert shape == patch_merge_3.init_out_size

    input_size = (H, W)
    dummy_input = torch.rand(B, C, H, W)
    # test stride and norm
    patch_merge_3 = PatchEmbed(
        in_channels=C,
        embed_dims=embed_dims,
        pad_to_stride=True,
        conv_type=None,
        kernel_size=kernel_size,
        stride=stride,
        padding=0,
        dilation=2,
        norm_cfg=dict(type='LN'),
        input_size=input_size)

    _, shape = patch_merge_3(dummy_input)
    # when input_size equal to real input
    # the out_size shoule be equal to `init_out_size`
    assert shape == patch_merge_3.init_out_size


def test_patch_merging():
    in_c = 3
    out_c = 4
    kernel_size = 3
    stride = 3
    padding = 1
    dilation = 1
    bias = False

    patch_merge = PatchMerging(
        in_channels=in_c,
        out_channels=out_c,
        pad_to_stride=False,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=bias)

    B, L, C = 1, 100, 3
    input_size = (10, 10)
    x = torch.rand(B, L, C)
    x_out, out_size = patch_merge(x, input_size)
    assert x_out.size() == (1, 16, 4)
    assert out_size == (4, 4)
    assert x_out.size(1) == out_size[0] * out_size[1]

    in_c = 4
    out_c = 5
    kernel_size = 6
    stride = 3
    padding = 2
    dilation = 2
    bias = False

    patch_merge = PatchMerging(
        in_channels=in_c,
        out_channels=out_c,
        pad_to_stride=False,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=bias)

    B, L, C = 1, 100, 4
    input_size = (10, 10)
    x = torch.rand(B, L, C)
    x_out, out_size = patch_merge(x, input_size)
    assert x_out.size() == (1, 4, 5)
    assert out_size == (2, 2)
    assert x_out.size(1) == out_size[0] * out_size[1]


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
