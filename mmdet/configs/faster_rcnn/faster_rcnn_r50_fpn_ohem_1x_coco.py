# Copyright (c) OpenMMLab. All rights reserved.
from mmengine import read_base

from mmdet.models.task_modules import OHEMSampler

with read_base():
    from .faster_rcnn_r50_fpn_1x_coco import *

model.update(dict(train_cfg=dict(rcnn=dict(sampler=dict(type=OHEMSampler)))))
