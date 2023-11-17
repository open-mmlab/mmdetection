# Copyright (c) OpenMMLab. All rights reserved.
from mmengine import read_base
from mmengine.model.weight_init import PretrainedInit

with read_base():
    from .faster_rcnn_r50_fpn_ms_3x_coco import *

model.update(
    dict(
        backbone=dict(
            depth=101,
            norm_cfg=dict(requires_grad=False),
            norm_eval=True,
            style='caffe',
            init_cfg=dict(
                type=PretrainedInit,
                checkpoint='open-mmlab://detectron2/resnet101_caffe'))))
