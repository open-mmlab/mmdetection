# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch
from mmengine.config import ConfigDict

from mmdet.models.layers.transformer import (AdaptivePadding,
                                             DetrTransformerDecoder,
                                             DetrTransformerEncoder,
                                             PatchEmbed, PatchMerging)


def test_adaptive_padding():

    for padding in ('same', 'corner'):
        kernel_size = 16
        stride = 16
        dilation = 1
        input = torch.rand(1, 1, 15, 17)
        pool = AdaptivePadding(
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            padding=padding)
        out = pool(input)
        # padding to divisible by 16
        assert (out.shape[2], out.shape[3]) == (16, 32)
        input = torch.rand(1, 1, 16, 17)
        out = pool(input)
        # padding to divisible by 16
        assert (out.shape[2], out.shape[3]) == (16, 32)

        kernel_size = (2, 2)
        stride = (2, 2)
        dilation = (1, 1)

        adap_pad = AdaptivePadding(
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            padding=padding)
        input = torch.rand(1, 1, 11, 13)
        out = adap_pad(input)
        # padding to divisible by 2
        assert (out.shape[2], out.shape[3]) == (12, 14)

        kernel_size = (2, 2)
        stride = (10, 10)
        dilation = (1, 1)

        adap_pad = AdaptivePadding(
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            padding=padding)
        input = torch.rand(1, 1, 10, 13)
        out = adap_pad(input)
        #  no padding
        assert (out.shape[2], out.shape[3]) == (10, 13)

        kernel_size = (11, 11)
        adap_pad = AdaptivePadding(
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            padding=padding)
        input = torch.rand(1, 1, 11, 13)
        out = adap_pad(input)
        #  all padding
        assert (out.shape[2], out.shape[3]) == (21, 21)

        # test padding as kernel is (7,9)
        input = torch.rand(1, 1, 11, 13)
        stride = (3, 4)
        kernel_size = (4, 5)
        dilation = (2, 2)
        # actually (7, 9)
        adap_pad = AdaptivePadding(
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            padding=padding)
        dilation_out = adap_pad(input)
        assert (dilation_out.shape[2], dilation_out.shape[3]) == (16, 21)
        kernel_size = (7, 9)
        dilation = (1, 1)
        adap_pad = AdaptivePadding(
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            padding=padding)
        kernel79_out = adap_pad(input)
        assert (kernel79_out.shape[2], kernel79_out.shape[3]) == (16, 21)
        assert kernel79_out.shape == dilation_out.shape

    # assert only support "same" "corner"
    with pytest.raises(AssertionError):
        AdaptivePadding(
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            padding=1)


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
        kernel_size=kernel_size,
        stride=stride,
        padding=0,
        dilation=1,
        norm_cfg=None)

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

    # test the init_out_size with nn.Unfold
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
        kernel_size=kernel_size,
        stride=stride,
        padding=0,
        dilation=2,
        norm_cfg=dict(type='LN'),
        input_size=input_size)

    _, shape = patch_merge_3(dummy_input)
    # when input_size equal to real input
    # the out_size should be equal to `init_out_size`
    assert shape == patch_merge_3.init_out_size

    input_size = (H, W)
    dummy_input = torch.rand(B, C, H, W)
    # test stride and norm
    patch_merge_3 = PatchEmbed(
        in_channels=C,
        embed_dims=embed_dims,
        kernel_size=kernel_size,
        stride=stride,
        padding=0,
        dilation=2,
        norm_cfg=dict(type='LN'),
        input_size=input_size)

    _, shape = patch_merge_3(dummy_input)
    # when input_size equal to real input
    # the out_size should be equal to `init_out_size`
    assert shape == patch_merge_3.init_out_size

    # test adap padding
    for padding in ('same', 'corner'):
        in_c = 2
        embed_dims = 3
        B = 2

        # test stride is 1
        input_size = (5, 5)
        kernel_size = (5, 5)
        stride = (1, 1)
        dilation = 1
        bias = False

        x = torch.rand(B, in_c, *input_size)
        patch_embed = PatchEmbed(
            in_channels=in_c,
            embed_dims=embed_dims,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias)

        x_out, out_size = patch_embed(x)
        assert x_out.size() == (B, 25, 3)
        assert out_size == (5, 5)
        assert x_out.size(1) == out_size[0] * out_size[1]

        # test kernel_size == stride
        input_size = (5, 5)
        kernel_size = (5, 5)
        stride = (5, 5)
        dilation = 1
        bias = False

        x = torch.rand(B, in_c, *input_size)
        patch_embed = PatchEmbed(
            in_channels=in_c,
            embed_dims=embed_dims,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias)

        x_out, out_size = patch_embed(x)
        assert x_out.size() == (B, 1, 3)
        assert out_size == (1, 1)
        assert x_out.size(1) == out_size[0] * out_size[1]

        # test kernel_size == stride
        input_size = (6, 5)
        kernel_size = (5, 5)
        stride = (5, 5)
        dilation = 1
        bias = False

        x = torch.rand(B, in_c, *input_size)
        patch_embed = PatchEmbed(
            in_channels=in_c,
            embed_dims=embed_dims,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias)

        x_out, out_size = patch_embed(x)
        assert x_out.size() == (B, 2, 3)
        assert out_size == (2, 1)
        assert x_out.size(1) == out_size[0] * out_size[1]

        # test different kernel_size with different stride
        input_size = (6, 5)
        kernel_size = (6, 2)
        stride = (6, 2)
        dilation = 1
        bias = False

        x = torch.rand(B, in_c, *input_size)
        patch_embed = PatchEmbed(
            in_channels=in_c,
            embed_dims=embed_dims,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias)

        x_out, out_size = patch_embed(x)
        assert x_out.size() == (B, 3, 3)
        assert out_size == (1, 3)
        assert x_out.size(1) == out_size[0] * out_size[1]


def test_patch_merging():

    # Test the model with int padding
    in_c = 3
    out_c = 4
    kernel_size = 3
    stride = 3
    padding = 1
    dilation = 1
    bias = False
    # test the case `pad_to_stride` is False
    patch_merge = PatchMerging(
        in_channels=in_c,
        out_channels=out_c,
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
    # assert out size is consistent with real output
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
    # assert out size is consistent with real output
    assert x_out.size(1) == out_size[0] * out_size[1]

    # Test with adaptive padding
    for padding in ('same', 'corner'):
        in_c = 2
        out_c = 3
        B = 2

        # test stride is 1
        input_size = (5, 5)
        kernel_size = (5, 5)
        stride = (1, 1)
        dilation = 1
        bias = False
        L = input_size[0] * input_size[1]

        x = torch.rand(B, L, in_c)
        patch_merge = PatchMerging(
            in_channels=in_c,
            out_channels=out_c,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias)

        x_out, out_size = patch_merge(x, input_size)
        assert x_out.size() == (B, 25, 3)
        assert out_size == (5, 5)
        assert x_out.size(1) == out_size[0] * out_size[1]

        # test kernel_size == stride
        input_size = (5, 5)
        kernel_size = (5, 5)
        stride = (5, 5)
        dilation = 1
        bias = False
        L = input_size[0] * input_size[1]

        x = torch.rand(B, L, in_c)
        patch_merge = PatchMerging(
            in_channels=in_c,
            out_channels=out_c,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias)

        x_out, out_size = patch_merge(x, input_size)
        assert x_out.size() == (B, 1, 3)
        assert out_size == (1, 1)
        assert x_out.size(1) == out_size[0] * out_size[1]

        # test kernel_size == stride
        input_size = (6, 5)
        kernel_size = (5, 5)
        stride = (5, 5)
        dilation = 1
        bias = False
        L = input_size[0] * input_size[1]

        x = torch.rand(B, L, in_c)
        patch_merge = PatchMerging(
            in_channels=in_c,
            out_channels=out_c,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias)

        x_out, out_size = patch_merge(x, input_size)
        assert x_out.size() == (B, 2, 3)
        assert out_size == (2, 1)
        assert x_out.size(1) == out_size[0] * out_size[1]

        # test different kernel_size with different stride
        input_size = (6, 5)
        kernel_size = (6, 2)
        stride = (6, 2)
        dilation = 1
        bias = False
        L = input_size[0] * input_size[1]

        x = torch.rand(B, L, in_c)
        patch_merge = PatchMerging(
            in_channels=in_c,
            out_channels=out_c,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias)

        x_out, out_size = patch_merge(x, input_size)
        assert x_out.size() == (B, 3, 3)
        assert out_size == (1, 3)
        assert x_out.size(1) == out_size[0] * out_size[1]


def test_detr_transformer_encoder_decoder():
    config = ConfigDict(
        num_layers=6,
        layer_cfg=dict(  # DetrTransformerDecoderLayer
            self_attn_cfg=dict(  # MultiheadAttention
                embed_dims=256,
                num_heads=8,
                dropout=0.1),
            cross_attn_cfg=dict(  # MultiheadAttention
                embed_dims=256,
                num_heads=8,
                dropout=0.1),
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,
                num_fcs=2,
                ffn_drop=0.1,
                act_cfg=dict(type='ReLU', inplace=True))))
    assert len(DetrTransformerDecoder(**config).layers) == 6
    assert DetrTransformerDecoder(**config)

    config = ConfigDict(
        dict(
            num_layers=6,
            layer_cfg=dict(  # DetrTransformerEncoderLayer
                self_attn_cfg=dict(  # MultiheadAttention
                    embed_dims=256,
                    num_heads=8,
                    dropout=0.1),
                ffn_cfg=dict(
                    embed_dims=256,
                    feedforward_channels=2048,
                    num_fcs=2,
                    ffn_drop=0.1,
                    act_cfg=dict(type='ReLU', inplace=True)))))
    assert len(DetrTransformerEncoder(**config).layers) == 6
    assert DetrTransformerEncoder(**config)
