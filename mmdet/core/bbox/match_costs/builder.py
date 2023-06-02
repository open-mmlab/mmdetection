# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.utils import Registry, build_from_cfg

MATCH_COST = Registry('Match Cost')


def build_match_cost(cfg, default_args=None):
    """Builder of IoU calculator."""
    return build_from_cfg(cfg, MATCH_COST, default_args)
