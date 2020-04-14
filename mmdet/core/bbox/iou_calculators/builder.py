from mmdet.utils import build_from_cfg
from .registry import IOU_CALCULATORS


def build_iou_calculator(cfg, default_args=None):
    return build_from_cfg(cfg, IOU_CALCULATORS, default_args)
