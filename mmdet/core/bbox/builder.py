# Copyright (c) OpenMMLab. All rights reserved.
import warnings

from mmdet.registry import TASK_UTILS

BBOX_ASSIGNERS = TASK_UTILS
BBOX_SAMPLERS = TASK_UTILS
BBOX_CODERS = TASK_UTILS


def build_assigner(cfg, **default_args):
    """Builder of box assigner."""
    warnings.warn('``build_assigner`` would be deprecated soon, please use '
                  '``mmdet.registry.TASK_UTILS.build()`` ')
    return TASK_UTILS.build(cfg, default_args=default_args)


def build_sampler(cfg, **default_args):
    """Builder of box sampler."""
    warnings.warn('``build_sampler`` would be deprecated soon, please use '
                  '``mmdet.registry.TASK_UTILS.build()`` ')
    return TASK_UTILS.build(cfg, default_args=default_args)


def build_bbox_coder(cfg, **default_args):
    """Builder of box coder."""
    warnings.warn('``build_sampler`` would be deprecated soon, please use '
                  '``mmdet.registry.TASK_UTILS.build()`` ')
    return TASK_UTILS.build(cfg, default_args=default_args)
