import torch

from mmdet.models.necks import FPN
from mmdet.models.necks.fpn_mapper import SemanticFPN


def test_semantic_fpn():
    s = 64
    in_channels = [8, 16, 32, 64]
    feat_sizes = [s // 2**i for i in range(4)]  # [64, 32, 16, 8]
    out_channels = 8

    fpn = FPN(
        in_channels=in_channels,
        out_channels=out_channels,
        start_level=0,
        num_outs=5)

    cat_coors = False
    cat_coors_level = 3
    return_list = False

    # FPN expects a multiple levels of features per image
    feats = [
        torch.rand(2, in_channels[i], feat_sizes[i], feat_sizes[i])
        for i in range(len(in_channels))
    ]

    # test normal settings used by SOLOv2
    semantic_fpn = SemanticFPN(
        in_channels=out_channels,
        feat_channels=out_channels,
        out_channels=out_channels,
        start_level=0,
        end_level=3,
        cat_coors=cat_coors,
        cat_coors_level=cat_coors_level,
        return_list=return_list,
        norm_cfg=dict(type='GN', num_groups=2, requires_grad=True))

    outs = semantic_fpn(fpn(feats))
    assert semantic_fpn.cat_coors == cat_coors
    assert semantic_fpn.convs_all_levels[
        cat_coors_level].conv0.conv.in_channels == out_channels
    assert outs.shape[-2] == s and outs.shape[-1] == s
    assert outs.shape[1] == out_channels

    cat_coors = True
    cat_coors_level = 2
    return_list = True
    # test normal settings used by SOLOv2
    semantic_fpn = SemanticFPN(
        in_channels=out_channels,
        feat_channels=out_channels,
        out_channels=out_channels,
        start_level=0,
        end_level=3,
        cat_coors=cat_coors,
        cat_coors_level=cat_coors_level,
        return_list=return_list,
        norm_cfg=dict(type='GN', num_groups=2, requires_grad=True))

    outs = semantic_fpn(fpn(feats))[0]
    assert semantic_fpn.cat_coors == cat_coors
    assert semantic_fpn.convs_all_levels[
        cat_coors_level].conv0.conv.in_channels == out_channels + 2
    assert semantic_fpn.conv_pred.conv.in_channels == out_channels
    assert outs.shape[-2] == s and outs.shape[-1] == s
    assert outs.shape[1] == out_channels
