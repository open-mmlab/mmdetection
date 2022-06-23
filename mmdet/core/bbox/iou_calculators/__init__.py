# Copyright (c) OpenMMLab. All rights reserved.
from .builder import build_iou_calculator
from .iou2d_calculator import (BboxOverlaps2D, alpha_bbox_overlaps,
                               bbox_overlaps)

__all__ = [
    'build_iou_calculator', 'BboxOverlaps2D', 'bbox_overlaps',
    'alpha_bbox_overlaps'
]
