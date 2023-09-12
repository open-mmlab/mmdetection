# Copyright (c) OpenMMLab. All rights reserved.

# Please refer to https://mmengine.readthedocs.io/en/latest/advanced_tutorials/config.html#a-pure-python-style-configuration-file-beta for more details. # noqa
# mmcv >= 2.0.1
# mmengine >= 0.8.0

from mmengine.config import read_base

with read_base():
    from .mask_rcnn_x101_32x4d_fpn_1x_coco import *

model = dict(
    # ResNeXt-101-32x8d model trained with Caffe2 at FB,
    # so the mean and std need to be changed.
    data_preprocessor=dict(
        mean=[103.530, 116.280, 123.675],
        std=[57.375, 57.120, 58.395],
        bgr_to_rgb=False),
    backbone=dict(
        type=ResNeXt,
        depth=101,
        groups=32,
        base_width=8,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type=BatchNorm2d, requires_grad=False),
        style='pytorch',
        init_cfg=dict(
            type=PretrainedInit,
            checkpoint='open-mmlab://detectron2/resnext101_32x8d')))
