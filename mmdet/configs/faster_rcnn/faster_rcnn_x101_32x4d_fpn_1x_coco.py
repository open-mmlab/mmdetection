# Copyright (c) OpenMMLab. All rights reserved.
from mmengine import read_base
from mmengine.model.weight_init import PretrainedInit
from torch.nn import BatchNorm2d

from mmdet.models import ResNeXt

with read_base():
    from .faster_rcnn_r50_fpn_1x_coco import *

model.update(
    dict(
        backbone=dict(
            type=ResNeXt,
            depth=101,
            groups=32,
            base_width=4,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=1,
            norm_cfg=dict(type=BatchNorm2d, requires_grad=True),
            style='pytorch',
            init_cfg=dict(
                type=PretrainedInit,
                checkpoint='open-mmlab://resnext101_32x4d'))))
