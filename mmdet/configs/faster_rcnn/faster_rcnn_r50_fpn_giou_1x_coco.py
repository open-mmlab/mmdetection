# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.config import read_base

from mmdet.models.losses.iou_loss import GIoULoss

with read_base():
    from .faster_rcnn_r50_fpn_1x_coco import *

model.update(
    dict(
        roi_head=dict(
            bbox_head=dict(
                reg_decoded_bbox=True,
                loss_bbox=dict(type=GIoULoss, loss_weight=10.0)))))
