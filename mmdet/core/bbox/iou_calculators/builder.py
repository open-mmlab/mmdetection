from mmdet.utils import build_from_cfg
from .registry import IOUCALCULATOR


def build_iou_calculator(cfg, default_args=None):
    return build_from_cfg(cfg, IOUCALCULATOR, default_args)
