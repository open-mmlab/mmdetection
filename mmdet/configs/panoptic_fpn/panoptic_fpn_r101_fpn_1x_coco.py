# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.config import read_base
from mmengine.model.weight_init import PretrainedInit

with read_base():
    from .panoptic_fpn_r50_fpn_1x_coco import *

model.update(
    dict(
        backbone=dict(
            depth=101,
            init_cfg=dict(
                type=PretrainedInit, checkpoint='torchvision://resnet101'))))
