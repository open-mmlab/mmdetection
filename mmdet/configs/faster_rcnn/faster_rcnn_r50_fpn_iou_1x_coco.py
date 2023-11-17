# Copyright (c) OpenMMLab. All rights reserved.
from mmengine import read_base

with read_base():
    from faster_rcnn_r50_fpn_1x_coco import *

model.update(
    dict(
        roi_head=dict(
            bbox_head=dict(
                reg_decoded_bbox=True,
                loss_bbox=dict(type='IoULoss', loss_weight=10.0)))))
