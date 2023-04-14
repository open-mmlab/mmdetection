if '_base_':
    from .._base_.models.faster_rcnn_r50_fpn import *
    from .._base_.datasets.coco_detection import *
    from .._base_.schedules.schedule_1x import *
    from .._base_.default_runtime import *
from mmdet.models.detectors.faster_rcnn import FasterRCNN
from mmdet.models.dense_heads.rpn_head import RPNHead
from mmdet.models.task_modules.prior_generators.anchor_generator import LegacyAnchorGenerator
from mmdet.models.task_modules.coders.legacy_delta_xywh_bbox_coder import LegacyDeltaXYWHBBoxCoder, LegacyDeltaXYWHBBoxCoder
from mmdet.models.losses.smooth_l1_loss import SmoothL1Loss, SmoothL1Loss
from mmdet.models.roi_heads.standard_roi_head import StandardRoIHead
from mmdet.models.roi_heads.roi_extractors.single_level_roi_extractor import SingleRoIExtractor

model.merge(
    dict(
        type=FasterRCNN,
        backbone=dict(
            init_cfg=dict(
                type='Pretrained', checkpoint='torchvision://resnet50')),
        rpn_head=dict(
            type=RPNHead,
            anchor_generator=dict(
                type=LegacyAnchorGenerator,
                center_offset=0.5,
                scales=[8],
                ratios=[0.5, 1.0, 2.0],
                strides=[4, 8, 16, 32, 64]),
            bbox_coder=dict(type=LegacyDeltaXYWHBBoxCoder),
            loss_bbox=dict(type=SmoothL1Loss, beta=1.0 / 9.0,
                           loss_weight=1.0)),
        roi_head=dict(
            type=StandardRoIHead,
            bbox_roi_extractor=dict(
                type=SingleRoIExtractor,
                roi_layer=dict(
                    type='RoIAlign',
                    output_size=7,
                    sampling_ratio=2,
                    aligned=False),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32]),
            bbox_head=dict(
                bbox_coder=dict(type=LegacyDeltaXYWHBBoxCoder),
                loss_bbox=dict(type=SmoothL1Loss, beta=1.0, loss_weight=1.0))),
        # model training and testing settings
        train_cfg=dict(
            rpn_proposal=dict(max_per_img=2000),
            rcnn=dict(assigner=dict(match_low_quality=True)))))
