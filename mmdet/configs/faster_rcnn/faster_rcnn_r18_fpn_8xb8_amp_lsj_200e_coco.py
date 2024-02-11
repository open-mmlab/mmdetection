# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.config import read_base
from mmengine.model.weight_init import PretrainedInit

with read_base():
    from .faster_rcnn_r50_fpn_8xb8_amp_lsj_200e_coco import *

model.update(
    dict(
        backbone=dict(
            depth=18,
            init_cfg=dict(
                type=PretrainedInit, checkpoint='torchvision://resnet18')),
        neck=dict(in_channels=[64, 128, 256, 512])))
