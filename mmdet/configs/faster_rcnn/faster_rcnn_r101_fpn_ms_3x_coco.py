# Copyright (c) OpenMMLab. All rights reserved.
from mmengine import read_base

with read_base():
    from .faster_rcnn_r50_fpn_ms_3x_coco import *

model.update(
    dict(
        backbone=dict(
            depth=101,
            init_cfg=dict(
                type='Pretrained', checkpoint='torchvision://resnet101'))))
