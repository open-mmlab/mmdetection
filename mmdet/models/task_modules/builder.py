# Copyright (c) OpenMMLab. All rights reserved.
import warnings

from mmdet.registry import TASK_UTILS

PRIOR_GENERATORS = TASK_UTILS
ANCHOR_GENERATORS = TASK_UTILS
BBOX_ASSIGNERS = TASK_UTILS
BBOX_SAMPLERS = TASK_UTILS
BBOX_CODERS = TASK_UTILS
MATCH_COSTS = TASK_UTILS
IOU_CALCULATORS = TASK_UTILS


def build_bbox_coder(cfg, **default_args):
    """Builder of box coder."""
    warnings.warn('``build_sampler`` would be deprecated soon, please use '
                  '``mmdet.registry.TASK_UTILS.build()`` ')
    return TASK_UTILS.build(cfg, default_args=default_args)


def build_iou_calculator(cfg, default_args=None):
    """Builder of IoU calculator."""
    warnings.warn(
        '``build_iou_calculator`` would be deprecated soon, please use '
        '``mmdet.registry.TASK_UTILS.build()`` ')
    return TASK_UTILS.build(cfg, default_args=default_args)


def build_match_cost(cfg, default_args=None):
    """Builder of IoU calculator."""
    warnings.warn('``build_match_cost`` would be deprecated soon, please use '
                  '``mmdet.registry.TASK_UTILS.build()`` ')
    return TASK_UTILS.build(cfg, default_args=default_args)


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


def build_prior_generator(cfg, default_args=None):
    warnings.warn(
        '``build_prior_generator`` would be deprecated soon, please use '
        '``mmdet.registry.TASK_UTILS.build()`` ')
    return TASK_UTILS.build(cfg, default_args=default_args)


def build_anchor_generator(cfg, default_args=None):
    warnings.warn(
        '``build_anchor_generator`` would be deprecated soon, please use '
        '``mmdet.registry.TASK_UTILS.build()`` ')
    return TASK_UTILS.build(cfg, default_args=default_args)
