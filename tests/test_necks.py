import pytest
import torch
from torch.nn.modules.batchnorm import _BatchNorm

from mmdet.models.necks import FPN


def test_fpn():
    """Tests fpn """
    s = 64
    in_channels = [8, 16, 32, 64]
    feat_sizes = [s // 2**i for i in range(4)]  # [64, 32, 16, 8]
    out_channels = 8
    # `num_outs` is not equal to len(in_channels) - start_level
    with pytest.raises(AssertionError):
        FPN(in_channels=in_channels,
            out_channels=out_channels,
            start_level=1,
            num_outs=2)

    # `end_level` is larger than len(in_channels) - 1
    with pytest.raises(AssertionError):
        FPN(in_channels=in_channels,
            out_channels=out_channels,
            start_level=1,
            end_level=4,
            num_outs=2)

    # `num_outs` is not equal to end_level - start_level
    with pytest.raises(AssertionError):
        FPN(in_channels=in_channels,
            out_channels=out_channels,
            start_level=1,
            end_level=3,
            num_outs=1)

    # Invalid `add_extra_convs` option
    with pytest.raises(AssertionError):
        FPN(in_channels=in_channels,
            out_channels=out_channels,
            start_level=1,
            add_extra_convs='on_xxx',
            num_outs=5)

    fpn_model = FPN(
        in_channels=in_channels,
        out_channels=out_channels,
        start_level=1,
        add_extra_convs=True,
        num_outs=5)

    # FPN expects a multiple levels of features per image
    feats = [
        torch.rand(1, in_channels[i], feat_sizes[i], feat_sizes[i])
        for i in range(len(in_channels))
    ]
    outs = fpn_model(feats)
    assert fpn_model.add_extra_convs == 'on_input'
    assert len(outs) == fpn_model.num_outs
    for i in range(fpn_model.num_outs):
        outs[i].shape[1] == out_channels
        outs[i].shape[2] == outs[i].shape[3] == s // (2**i)

    # Tests for fpn with no extra convs (pooling is used instead)
    fpn_model = FPN(
        in_channels=in_channels,
        out_channels=out_channels,
        start_level=1,
        add_extra_convs=False,
        num_outs=5)
    outs = fpn_model(feats)
    assert len(outs) == fpn_model.num_outs
    assert not fpn_model.add_extra_convs
    for i in range(fpn_model.num_outs):
        outs[i].shape[1] == out_channels
        outs[i].shape[2] == outs[i].shape[3] == s // (2**i)

    # Tests for fpn with lateral bns
    fpn_model = FPN(
        in_channels=in_channels,
        out_channels=out_channels,
        start_level=1,
        add_extra_convs=True,
        no_norm_on_lateral=False,
        norm_cfg=dict(type='BN', requires_grad=True),
        num_outs=5)
    outs = fpn_model(feats)
    assert len(outs) == fpn_model.num_outs
    assert fpn_model.add_extra_convs == 'on_input'
    for i in range(fpn_model.num_outs):
        outs[i].shape[1] == out_channels
        outs[i].shape[2] == outs[i].shape[3] == s // (2**i)
    bn_exist = False
    for m in fpn_model.modules():
        if isinstance(m, _BatchNorm):
            bn_exist = True
    assert bn_exist

    # Bilinear upsample
    fpn_model = FPN(
        in_channels=in_channels,
        out_channels=out_channels,
        start_level=1,
        add_extra_convs=True,
        upsample_cfg=dict(mode='bilinear', align_corners=True),
        num_outs=5)
    fpn_model(feats)
    outs = fpn_model(feats)
    assert len(outs) == fpn_model.num_outs
    assert fpn_model.add_extra_convs == 'on_input'
    for i in range(fpn_model.num_outs):
        outs[i].shape[1] == out_channels
        outs[i].shape[2] == outs[i].shape[3] == s // (2**i)

    # Scale factor instead of fixed upsample size upsample
    fpn_model = FPN(
        in_channels=in_channels,
        out_channels=out_channels,
        start_level=1,
        add_extra_convs=True,
        upsample_cfg=dict(scale_factor=2),
        num_outs=5)
    outs = fpn_model(feats)
    assert len(outs) == fpn_model.num_outs
    for i in range(fpn_model.num_outs):
        outs[i].shape[1] == out_channels
        outs[i].shape[2] == outs[i].shape[3] == s // (2**i)

    # Extra convs source is 'inputs'
    fpn_model = FPN(
        in_channels=in_channels,
        out_channels=out_channels,
        add_extra_convs='on_input',
        start_level=1,
        num_outs=5)
    assert fpn_model.add_extra_convs == 'on_input'
    outs = fpn_model(feats)
    assert len(outs) == fpn_model.num_outs
    for i in range(fpn_model.num_outs):
        outs[i].shape[1] == out_channels
        outs[i].shape[2] == outs[i].shape[3] == s // (2**i)

    # Extra convs source is 'laterals'
    fpn_model = FPN(
        in_channels=in_channels,
        out_channels=out_channels,
        add_extra_convs='on_lateral',
        start_level=1,
        num_outs=5)
    assert fpn_model.add_extra_convs == 'on_lateral'
    outs = fpn_model(feats)
    assert len(outs) == fpn_model.num_outs
    for i in range(fpn_model.num_outs):
        outs[i].shape[1] == out_channels
        outs[i].shape[2] == outs[i].shape[3] == s // (2**i)

    # Extra convs source is 'outputs'
    fpn_model = FPN(
        in_channels=in_channels,
        out_channels=out_channels,
        add_extra_convs='on_output',
        start_level=1,
        num_outs=5)
    assert fpn_model.add_extra_convs == 'on_output'
    outs = fpn_model(feats)
    assert len(outs) == fpn_model.num_outs
    for i in range(fpn_model.num_outs):
        outs[i].shape[1] == out_channels
        outs[i].shape[2] == outs[i].shape[3] == s // (2**i)

    # extra_convs_on_inputs=False is equal to extra convs source is 'on_output'
    fpn_model = FPN(
        in_channels=in_channels,
        out_channels=out_channels,
        add_extra_convs=True,
        extra_convs_on_inputs=False,
        start_level=1,
        num_outs=5,
    )
    assert fpn_model.add_extra_convs == 'on_output'
    outs = fpn_model(feats)
    assert len(outs) == fpn_model.num_outs
    for i in range(fpn_model.num_outs):
        outs[i].shape[1] == out_channels
        outs[i].shape[2] == outs[i].shape[3] == s // (2**i)

    # extra_convs_on_inputs=True is equal to extra convs source is 'on_input'
    fpn_model = FPN(
        in_channels=in_channels,
        out_channels=out_channels,
        add_extra_convs=True,
        extra_convs_on_inputs=True,
        start_level=1,
        num_outs=5,
    )
    assert fpn_model.add_extra_convs == 'on_input'
    outs = fpn_model(feats)
    assert len(outs) == fpn_model.num_outs
    for i in range(fpn_model.num_outs):
        outs[i].shape[1] == out_channels
        outs[i].shape[2] == outs[i].shape[3] == s // (2**i)
