# Copyright (c) OpenMMLab. All rights reserved.
import warnings

from mmdet.registry import MODELS

BACKBONES = MODELS
NECKS = MODELS
ROI_EXTRACTORS = MODELS
SHARED_HEADS = MODELS
HEADS = MODELS
LOSSES = MODELS
DETECTORS = MODELS


def build_backbone(cfg):
    """Build backbone."""
    warnings.warn('``build_backbone`` would be deprecated soon, please use '
                  '``mmdet.registry.MODELS.build()`` ')
    return BACKBONES.build(cfg)


def build_neck(cfg):
    """Build neck."""
    warnings.warn('``build_neck`` would be deprecated soon, please use '
                  '``mmdet.registry.MODELS.build()`` ')
    return NECKS.build(cfg)


def build_roi_extractor(cfg):
    """Build roi extractor."""
    warnings.warn(
        '``build_roi_extractor`` would be deprecated soon, please use '
        '``mmdet.registry.MODELS.build()`` ')
    return ROI_EXTRACTORS.build(cfg)


def build_shared_head(cfg):
    """Build shared head."""
    warnings.warn('``build_shared_head`` would be deprecated soon, please use '
                  '``mmdet.registry.MODELS.build()`` ')
    return SHARED_HEADS.build(cfg)


def build_head(cfg):
    """Build head."""
    warnings.warn('``build_head`` would be deprecated soon, please use '
                  '``mmdet.registry.MODELS.build()`` ')
    return HEADS.build(cfg)


def build_loss(cfg):
    """Build loss."""
    warnings.warn('``build_loss`` would be deprecated soon, please use '
                  '``mmdet.registry.MODELS.build()`` ')
    return LOSSES.build(cfg)


def build_detector(cfg, train_cfg=None, test_cfg=None):
    """Build detector."""
    if train_cfg is not None or test_cfg is not None:
        warnings.warn(
            'train_cfg and test_cfg is deprecated, '
            'please specify them in model', UserWarning)
    assert cfg.get('train_cfg') is None or train_cfg is None, \
        'train_cfg specified in both outer field and model field '
    assert cfg.get('test_cfg') is None or test_cfg is None, \
        'test_cfg specified in both outer field and model field '
    return DETECTORS.build(
        cfg, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg))
