if '_base_':
    from ..cascade_rcnn.cascade_rcnn_r50_fpn_1x_coco import *
from mmdet.models.backbones.resnest import ResNeSt
from mmdet.models.roi_heads.bbox_heads.convfc_bbox_head import Shared4Conv1FCBBoxHead
from mmdet.models.task_modules.coders.delta_xywh_bbox_coder import DeltaXYWHBBoxCoder
from mmdet.models.losses.cross_entropy_loss import CrossEntropyLoss
from mmdet.models.losses.smooth_l1_loss import SmoothL1Loss
from mmdet.models.roi_heads.bbox_heads.convfc_bbox_head import Shared4Conv1FCBBoxHead
from mmdet.models.task_modules.coders.delta_xywh_bbox_coder import DeltaXYWHBBoxCoder
from mmdet.models.losses.cross_entropy_loss import CrossEntropyLoss
from mmdet.models.losses.smooth_l1_loss import SmoothL1Loss
from mmdet.models.roi_heads.bbox_heads.convfc_bbox_head import Shared4Conv1FCBBoxHead
from mmdet.models.task_modules.coders.delta_xywh_bbox_coder import DeltaXYWHBBoxCoder
from mmdet.models.losses.cross_entropy_loss import CrossEntropyLoss
from mmdet.models.losses.smooth_l1_loss import SmoothL1Loss
from mmcv.transforms.loading import LoadImageFromFile
from mmdet.datasets.transforms.loading import LoadAnnotations
from mmcv.transforms.processing import RandomResize
from mmdet.datasets.transforms.transforms import RandomFlip
from mmdet.datasets.transforms.formatting import PackDetInputs

norm_cfg = dict(type='SyncBN', requires_grad=True)
model.update(
    dict(
        # use ResNeSt img_norm
        data_preprocessor=dict(
            mean=[123.68, 116.779, 103.939],
            std=[58.393, 57.12, 57.375],
            bgr_to_rgb=True),
        backbone=dict(
            type=ResNeSt,
            stem_channels=64,
            depth=50,
            radix=2,
            reduction_factor=4,
            avg_down_stride=True,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=1,
            norm_cfg=norm_cfg,
            norm_eval=False,
            style='pytorch',
            init_cfg=dict(
                type='Pretrained', checkpoint='open-mmlab://resnest50')),
        roi_head=dict(
            bbox_head=[
                dict(
                    type=Shared4Conv1FCBBoxHead,
                    in_channels=256,
                    conv_out_channels=256,
                    fc_out_channels=1024,
                    norm_cfg=norm_cfg,
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
                    type=Shared4Conv1FCBBoxHead,
                    in_channels=256,
                    conv_out_channels=256,
                    fc_out_channels=1024,
                    norm_cfg=norm_cfg,
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
                    type=Shared4Conv1FCBBoxHead,
                    in_channels=256,
                    conv_out_channels=256,
                    fc_out_channels=1024,
                    norm_cfg=norm_cfg,
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
            ], )))

train_pipeline = [
    dict(type=LoadImageFromFile, backend_args={{_base_.backend_args}}),
    dict(type=LoadAnnotations, with_bbox=True),
    dict(type=RandomResize, scale=[(1333, 640), (1333, 800)], keep_ratio=True),
    dict(type=RandomFlip, prob=0.5),
    dict(type=PackDetInputs)
]

train_dataloader.update(dict(dataset=dict(pipeline=train_pipeline)))
