# Copyright (c) OpenMMLab. All rights reserved.
from mmengine import read_base
from mmengine.model.weight_init import PretrainedInit
from torch.optim.adamw import AdamW

with read_base():
    from .._base_.datasets.coco_detection import *
    from .._base_.default_runtime import *
    from .._base_.models.faster_rcnn_r50_fpn import *
    from .._base_.schedules.schedule_1x import *

checkpoint = 'https://download.pytorch.org/models/resnet50-11ad3fa6.pth'
model.update(
    dict(
        backbone=dict(
            init_cfg=dict(type=PretrainedInit, checkpoint=checkpoint))))

# `lr` and `weight_decay` have been searched to be optimal.
optim_wrapper.update(
    dict(
        optimizer=dict(_delete_=True, type=AdamW, lr=0.0001, weight_decay=0.1),
        paramwise_cfg=dict(norm_decay_mult=0., bypass_duplicate=True)))
