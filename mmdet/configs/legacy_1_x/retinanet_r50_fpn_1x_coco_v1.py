if '_base_':
    from .._base_.models.retinanet_r50_fpn import *
    from .._base_.datasets.coco_detection import *
    from .._base_.schedules.schedule_1x import *
    from .._base_.default_runtime import *
from mmdet.models.dense_heads.retina_head import RetinaHead
from mmdet.models.task_modules.prior_generators.anchor_generator import LegacyAnchorGenerator
from mmdet.models.task_modules.coders.legacy_delta_xywh_bbox_coder import LegacyDeltaXYWHBBoxCoder
from mmdet.models.losses.smooth_l1_loss import SmoothL1Loss

model.merge(
    dict(
        bbox_head=dict(
            type=RetinaHead,
            anchor_generator=dict(
                type=LegacyAnchorGenerator,
                center_offset=0.5,
                octave_base_scale=4,
                scales_per_octave=3,
                ratios=[0.5, 1.0, 2.0],
                strides=[8, 16, 32, 64, 128]),
            bbox_coder=dict(type=LegacyDeltaXYWHBBoxCoder),
            loss_bbox=dict(type=SmoothL1Loss, beta=0.11, loss_weight=1.0))))
