import mmcv
import torch

from mmdet.models.roi_heads.roi_extractors import SumGenericRoiExtractor


def test_groie():
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

    groie = SumGenericRoiExtractor(**cfg)

    feats = (
        torch.rand((1, 256, 200, 336)),
        torch.rand((1, 256, 100, 168)),
        torch.rand((1, 256, 50, 84)),
        torch.rand((1, 256, 25, 42)),
    )

    rois = torch.tensor([[0.0000, 587.8285, 52.1405, 886.2484, 341.5644]])

    res = groie(feats, rois)
    assert res.shape == torch.Size([1, 256, 7, 7])
