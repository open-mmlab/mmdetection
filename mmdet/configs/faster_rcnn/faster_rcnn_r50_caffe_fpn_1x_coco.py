# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.config import read_base
from mmengine.model.weight_init import PretrainedInit

with read_base():
    from .faster_rcnn_r50_fpn_1x_coco import *

model.update(
    dict(
        data_preprocessor=dict(
            type=DetDataPreprocessor,
            mean=[103.530, 116.280, 123.675],
            std=[1.0, 1.0, 1.0],
            bgr_to_rgb=False,
            pad_size_divisor=32),
        backbone=dict(
            norm_cfg=dict(requires_grad=False),
            norm_eval=True,
            style='caffe',
            init_cfg=dict(
                type=PretrainedInit,
                checkpoint='open-mmlab://detectron2/resnet50_caffe'))))
