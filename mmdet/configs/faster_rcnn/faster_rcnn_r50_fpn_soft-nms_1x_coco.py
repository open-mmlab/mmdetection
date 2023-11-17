# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.ops import soft_nms
from mmengine.config import read_base

with read_base():
    from .._base_.datasets.coco_detection import *
    from .._base_.default_runtime import *
    from .._base_.models.faster_rcnn_r50_fpn import *
    from .._base_.schedules.schedule_1x import *

model.update(
    dict(
        test_cfg=dict(
            rcnn=dict(
                score_thr=0.05,
                nms=dict(type=soft_nms, iou_threshold=0.5),
                max_per_img=100))))
