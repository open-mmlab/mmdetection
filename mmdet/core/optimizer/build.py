# Copyright (c) OpenMMLab. All rights reserved.
import copy

from mmcv import Registry, build_from_cfg
from mmcv.runner import OPTIMIZER_BUILDERS as MMCV_OPTIMIZER_BUILDERS

OPTIMIZER_BUILDERS = Registry(
    'optimizer builder', parent=MMCV_OPTIMIZER_BUILDERS)


def build_optimizer_constructor(cfg):
    return build_from_cfg(cfg, OPTIMIZER_BUILDERS)


def build_optimizer(model, cfg):
    optimizer_cfg = copy.deepcopy(cfg)
    constructor_type = optimizer_cfg.pop('constructor',
                                         'DefaultOptimizerConstructor')
    paramwise_cfg = optimizer_cfg.pop('paramwise_cfg', None)
    optim_constructor = build_optimizer_constructor(
        dict(
            type=constructor_type,
            optimizer_cfg=optimizer_cfg,
            paramwise_cfg=paramwise_cfg))
    optimizer = optim_constructor(model)
    return optimizer
