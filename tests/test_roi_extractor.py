import mmcv
import pytest
import torch

from mmdet.models.roi_heads.roi_extractors import GenericRoiExtractor


def test_groie():
    # test with pre/post
    cfg = mmcv.Config(
        dict(
            roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32],
            pre_cfg=dict(
                type='ConvModule',
                in_channels=256,
                out_channels=256,
                kernel_size=5,
                padding=2,
                inplace=False,
            ),
            post_cfg=dict(
                type='ConvModule',
                in_channels=256,
                out_channels=256,
                kernel_size=5,
                padding=2,
                inplace=False)))

    groie = GenericRoiExtractor(**cfg)

    feats = (
        torch.rand((1, 256, 200, 336)),
        torch.rand((1, 256, 100, 168)),
        torch.rand((1, 256, 50, 84)),
        torch.rand((1, 256, 25, 42)),
    )

    rois = torch.tensor([[0.0000, 587.8285, 52.1405, 886.2484, 341.5644]])

    res = groie(feats, rois)
    assert res.shape == torch.Size([1, 256, 7, 7])

    # test w.o. pre/post
    cfg = mmcv.Config(
        dict(
            roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]))

    groie = GenericRoiExtractor(**cfg)

    feats = (
        torch.rand((1, 256, 200, 336)),
        torch.rand((1, 256, 100, 168)),
        torch.rand((1, 256, 50, 84)),
        torch.rand((1, 256, 25, 42)),
    )

    rois = torch.tensor([[0.0000, 587.8285, 52.1405, 886.2484, 341.5644]])

    res = groie(feats, rois)
    assert res.shape == torch.Size([1, 256, 7, 7])

    # test w.o. pre/post concat
    cfg = mmcv.Config(
        dict(
            aggregate_type='concat',
            roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),
            out_channels=256 * 4,
            featmap_strides=[4, 8, 16, 32]))

    groie = GenericRoiExtractor(**cfg)

    feats = (
        torch.rand((1, 256, 200, 336)),
        torch.rand((1, 256, 100, 168)),
        torch.rand((1, 256, 50, 84)),
        torch.rand((1, 256, 25, 42)),
    )

    rois = torch.tensor([[0.0000, 587.8285, 52.1405, 886.2484, 341.5644]])

    res = groie(feats, rois)
    assert res.shape == torch.Size([1, 1024, 7, 7])

    # test not supported aggregate method
    with pytest.raises(AssertionError):
        cfg = mmcv.Config(
            dict(
                aggregate_type='not support',
                roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),
                out_channels=1024,
                featmap_strides=[4, 8, 16, 32]))
        _ = GenericRoiExtractor(**cfg)

    # test concat channels number
    cfg = mmcv.Config(
        dict(
            aggregate_type='concat',
            roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),
            out_channels=256 * 5,  # 256*5 != 256*4
            featmap_strides=[4, 8, 16, 32]))

    groie = GenericRoiExtractor(**cfg)

    feats = (
        torch.rand((1, 256, 200, 336)),
        torch.rand((1, 256, 100, 168)),
        torch.rand((1, 256, 50, 84)),
        torch.rand((1, 256, 25, 42)),
    )

    rois = torch.tensor([[0.0000, 587.8285, 52.1405, 886.2484, 341.5644]])

    with pytest.raises(AssertionError):
        _ = groie(feats, rois)
