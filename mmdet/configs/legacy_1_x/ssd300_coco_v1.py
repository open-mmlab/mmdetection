if '_base_':
    from .._base_.models.ssd300 import *
    from .._base_.datasets.coco_detection import *
    from .._base_.schedules.schedule_2x import *
    from .._base_.default_runtime import *
from mmdet.models.dense_heads.ssd_head import SSDHead
from mmdet.models.task_modules.prior_generators.anchor_generator import LegacySSDAnchorGenerator
from mmdet.models.task_modules.coders.legacy_delta_xywh_bbox_coder import LegacyDeltaXYWHBBoxCoder
# model settings
input_size = 300
model.merge(
    dict(
        bbox_head=dict(
            type=SSDHead,
            anchor_generator=dict(
                type=LegacySSDAnchorGenerator,
                scale_major=False,
                input_size=input_size,
                basesize_ratio_range=(0.15, 0.9),
                strides=[8, 16, 32, 64, 100, 300],
                ratios=[[2], [2, 3], [2, 3], [2, 3], [2], [2]]),
            bbox_coder=dict(
                type=LegacyDeltaXYWHBBoxCoder,
                target_means=[.0, .0, .0, .0],
                target_stds=[0.1, 0.1, 0.2, 0.2]))))
