import mmcv
from mmcv import torchpack
from torch import nn

from . import (backbones, necks, roi_extractors, rpn_heads, bbox_heads,
               mask_heads)

__all__ = [
    'build_backbone', 'build_neck', 'build_rpn_head', 'build_roi_extractor',
    'build_bbox_head', 'build_mask_head'
]


def _build_module(cfg, parrent=None):
    return cfg if isinstance(cfg, nn.Module) else torchpack.obj_from_dict(
        cfg, parrent)


def build(cfg, parrent=None):
    if isinstance(cfg, list):
        modules = [_build_module(cfg_, parrent) for cfg_ in cfg]
        return nn.Sequential(*modules)
    else:
        return _build_module(cfg, parrent)


def build_backbone(cfg):
    return build(cfg, backbones)


def build_neck(cfg):
    return build(cfg, necks)


def build_rpn_head(cfg):
    return build(cfg, rpn_heads)


def build_roi_extractor(cfg):
    return build(cfg, roi_extractors)


def build_bbox_head(cfg):
    return build(cfg, bbox_heads)


def build_mask_head(cfg):
    return build(cfg, mask_heads)
