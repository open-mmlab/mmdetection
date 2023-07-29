# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.config import read_base

with read_base():
    from .htc_x101_32x4d_fpn_16xb1_20e_coco import *

model.update(
    dict(
        backbone=dict(
            type=ResNeXt,
            groups=64,
            init_cfg=dict(
                type=PretrainedInit,
                checkpoint='open-mmlab://resnext101_64x4d'))))
