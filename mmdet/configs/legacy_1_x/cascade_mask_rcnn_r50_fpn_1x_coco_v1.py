if '_base_':
    from .._base_.models.cascade_mask_rcnn_r50_fpn import *
    from .._base_.datasets.coco_instance import *
    from .._base_.schedules.schedule_1x import *
    from .._base_.default_runtime import *
from mmdet.models.detectors.cascade_rcnn import CascadeRCNN
from mmdet.models.backbones.resnet import ResNet
from mmdet.models.necks.fpn import FPN
from mmdet.models.task_modules.prior_generators.anchor_generator import LegacyAnchorGenerator
from mmdet.models.task_modules.coders.legacy_delta_xywh_bbox_coder import LegacyDeltaXYWHBBoxCoder, LegacyDeltaXYWHBBoxCoder, LegacyDeltaXYWHBBoxCoder, LegacyDeltaXYWHBBoxCoder
from mmdet.models.roi_heads.roi_extractors.single_level_roi_extractor import SingleRoIExtractor, SingleRoIExtractor
from mmdet.models.roi_heads.bbox_heads.convfc_bbox_head import Shared2FCBBoxHead, Shared2FCBBoxHead, Shared2FCBBoxHead

model.merge(
    dict(
        type=CascadeRCNN,
        backbone=dict(
            type=ResNet,
            depth=50,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=True,
            style='pytorch',
            init_cfg=dict(
                type='Pretrained', checkpoint='torchvision://resnet50')),
        neck=dict(
            type=FPN,
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            num_outs=5),
        rpn_head=dict(
            anchor_generator=dict(
                type=LegacyAnchorGenerator, center_offset=0.5),
            bbox_coder=dict(
                type=LegacyDeltaXYWHBBoxCoder,
                target_means=[.0, .0, .0, .0],
                target_stds=[1.0, 1.0, 1.0, 1.0])),
        roi_head=dict(
            bbox_roi_extractor=dict(
                type=SingleRoIExtractor,
                roi_layer=dict(
                    type='RoIAlign',
                    output_size=7,
                    sampling_ratio=2,
                    aligned=False)),
            bbox_head=[
                dict(
                    type=Shared2FCBBoxHead,
                    reg_class_agnostic=True,
                    in_channels=256,
                    fc_out_channels=1024,
                    roi_feat_size=7,
                    num_classes=80,
                    bbox_coder=dict(
                        type=LegacyDeltaXYWHBBoxCoder,
                        target_means=[0., 0., 0., 0.],
                        target_stds=[0.1, 0.1, 0.2, 0.2])),
                dict(
                    type=Shared2FCBBoxHead,
                    reg_class_agnostic=True,
                    in_channels=256,
                    fc_out_channels=1024,
                    roi_feat_size=7,
                    num_classes=80,
                    bbox_coder=dict(
                        type=LegacyDeltaXYWHBBoxCoder,
                        target_means=[0., 0., 0., 0.],
                        target_stds=[0.05, 0.05, 0.1, 0.1])),
                dict(
                    type=Shared2FCBBoxHead,
                    reg_class_agnostic=True,
                    in_channels=256,
                    fc_out_channels=1024,
                    roi_feat_size=7,
                    num_classes=80,
                    bbox_coder=dict(
                        type=LegacyDeltaXYWHBBoxCoder,
                        target_means=[0., 0., 0., 0.],
                        target_stds=[0.033, 0.033, 0.067, 0.067])),
            ],
            mask_roi_extractor=dict(
                type=SingleRoIExtractor,
                roi_layer=dict(
                    type='RoIAlign',
                    output_size=14,
                    sampling_ratio=2,
                    aligned=False)))))
