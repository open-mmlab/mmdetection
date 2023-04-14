if '_base_':
    from ..htc.htc_r50_fpn_1x_coco import *
from mmdet.models.detectors.scnet import SCNet
from mmdet.models.roi_heads.scnet_roi_head import SCNetRoIHead
from mmdet.models.roi_heads.roi_extractors.single_level_roi_extractor import SingleRoIExtractor, SingleRoIExtractor, SingleRoIExtractor
from mmdet.models.roi_heads.bbox_heads.scnet_bbox_head import SCNetBBoxHead, SCNetBBoxHead, SCNetBBoxHead
from mmdet.models.task_modules.coders.delta_xywh_bbox_coder import DeltaXYWHBBoxCoder, DeltaXYWHBBoxCoder, DeltaXYWHBBoxCoder
from mmdet.models.losses.cross_entropy_loss import CrossEntropyLoss, CrossEntropyLoss, CrossEntropyLoss, CrossEntropyLoss, CrossEntropyLoss
from mmdet.models.losses.smooth_l1_loss import SmoothL1Loss, SmoothL1Loss, SmoothL1Loss
from mmdet.models.roi_heads.mask_heads.scnet_mask_head import SCNetMaskHead
from mmdet.models.roi_heads.mask_heads.scnet_semantic_head import SCNetSemanticHead
from mmdet.models.roi_heads.mask_heads.global_context_head import GlobalContextHead
from mmdet.models.roi_heads.mask_heads.feature_relay_head import FeatureRelayHead
from mmcv.transforms.loading import LoadImageFromFile
from mmcv.transforms.processing import MultiScaleFlipAug, Normalize
from mmdet.datasets.transforms.transforms import Resize, RandomFlip, Pad
from mmdet.datasets.transforms.formatting import ImageToTensor
# model settings
model.merge(
    dict(
        type=SCNet,
        roi_head=dict(
            _delete_=True,
            type=SCNetRoIHead,
            num_stages=3,
            stage_loss_weights=[1, 0.5, 0.25],
            bbox_roi_extractor=dict(
                type=SingleRoIExtractor,
                roi_layer=dict(
                    type='RoIAlign', output_size=7, sampling_ratio=0),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32]),
            bbox_head=[
                dict(
                    type=SCNetBBoxHead,
                    num_shared_fcs=2,
                    in_channels=256,
                    fc_out_channels=1024,
                    roi_feat_size=7,
                    num_classes=80,
                    bbox_coder=dict(
                        type=DeltaXYWHBBoxCoder,
                        target_means=[0., 0., 0., 0.],
                        target_stds=[0.1, 0.1, 0.2, 0.2]),
                    reg_class_agnostic=True,
                    loss_cls=dict(
                        type=CrossEntropyLoss,
                        use_sigmoid=False,
                        loss_weight=1.0),
                    loss_bbox=dict(
                        type=SmoothL1Loss, beta=1.0, loss_weight=1.0)),
                dict(
                    type=SCNetBBoxHead,
                    num_shared_fcs=2,
                    in_channels=256,
                    fc_out_channels=1024,
                    roi_feat_size=7,
                    num_classes=80,
                    bbox_coder=dict(
                        type=DeltaXYWHBBoxCoder,
                        target_means=[0., 0., 0., 0.],
                        target_stds=[0.05, 0.05, 0.1, 0.1]),
                    reg_class_agnostic=True,
                    loss_cls=dict(
                        type=CrossEntropyLoss,
                        use_sigmoid=False,
                        loss_weight=1.0),
                    loss_bbox=dict(
                        type=SmoothL1Loss, beta=1.0, loss_weight=1.0)),
                dict(
                    type=SCNetBBoxHead,
                    num_shared_fcs=2,
                    in_channels=256,
                    fc_out_channels=1024,
                    roi_feat_size=7,
                    num_classes=80,
                    bbox_coder=dict(
                        type=DeltaXYWHBBoxCoder,
                        target_means=[0., 0., 0., 0.],
                        target_stds=[0.033, 0.033, 0.067, 0.067]),
                    reg_class_agnostic=True,
                    loss_cls=dict(
                        type=CrossEntropyLoss,
                        use_sigmoid=False,
                        loss_weight=1.0),
                    loss_bbox=dict(
                        type=SmoothL1Loss, beta=1.0, loss_weight=1.0))
            ],
            mask_roi_extractor=dict(
                type=SingleRoIExtractor,
                roi_layer=dict(
                    type='RoIAlign', output_size=14, sampling_ratio=0),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32]),
            mask_head=dict(
                type=SCNetMaskHead,
                num_convs=12,
                in_channels=256,
                conv_out_channels=256,
                num_classes=80,
                conv_to_res=True,
                loss_mask=dict(
                    type=CrossEntropyLoss, use_mask=True, loss_weight=1.0)),
            semantic_roi_extractor=dict(
                type=SingleRoIExtractor,
                roi_layer=dict(
                    type='RoIAlign', output_size=14, sampling_ratio=0),
                out_channels=256,
                featmap_strides=[8]),
            semantic_head=dict(
                type=SCNetSemanticHead,
                num_ins=5,
                fusion_level=1,
                seg_scale_factor=1 / 8,
                num_convs=4,
                in_channels=256,
                conv_out_channels=256,
                num_classes=183,
                loss_seg=dict(
                    type=CrossEntropyLoss, ignore_index=255, loss_weight=0.2),
                conv_to_res=True),
            glbctx_head=dict(
                type=GlobalContextHead,
                num_convs=4,
                in_channels=256,
                conv_out_channels=256,
                num_classes=80,
                loss_weight=3.0,
                conv_to_res=True),
            feat_relay_head=dict(
                type=FeatureRelayHead,
                in_channels=1024,
                out_conv_channels=256,
                roi_feat_size=7,
                scale_factor=2))))

# TODO
# uncomment below code to enable test time augmentations
# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# test_pipeline = [
#     dict(type=LoadImageFromFile),
#     dict(
#         type=MultiScaleFlipAug,
#         img_scale=[(600, 900), (800, 1200), (1000, 1500), (1200, 1800),
#                    (1400, 2100)],
#         flip=True,
#         transforms=[
#             dict(type=Resize, keep_ratio=True),
#             dict(type=RandomFlip, flip_ratio=0.5),
#             dict(type=Normalize, **img_norm_cfg),
#             dict(type=Pad, size_divisor=32),
#             dict(type=ImageToTensor, keys=['img']),
#             dict(type='Collect', keys=['img']),
#         ])
# ]
# data = dict(
#     val=dict(pipeline=test_pipeline),
#     test=dict(pipeline=test_pipeline))
