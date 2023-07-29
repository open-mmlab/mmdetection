# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.config import read_base

with read_base():
    from .htc_r50_fpn_20e_coco import *

model.update(
    dict(
        type=HybridTaskCascade,
        backbone=dict(
            type=ResNet,
            depth=101,
            init_cfg=dict(
                type=PretrainedInit, checkpoint='torchvision://resnet101'))))
