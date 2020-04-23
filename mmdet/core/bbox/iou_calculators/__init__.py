from .iou2d_calculator import BboxOverlaps2D, bbox_overlaps
from .registry import build_iou_calculator

__all__ = ['build_iou_calculator', 'BboxOverlaps2D', 'bbox_overlaps']
