# Copyright (c) OpenMMLab. All rights reserved.
import warnings

from mmdet.registry import TASK_UTILS

IOU_CALCULATORS = TASK_UTILS


def build_iou_calculator(cfg, default_args=None):
    """Builder of IoU calculator."""
    warnings.warn(
        '``build_iou_calculator`` would be deprecated soon, please use '
        '``mmdet.registry.TASK_UTILS.build()`` ')
    return TASK_UTILS.build(cfg, default_args=default_args)
