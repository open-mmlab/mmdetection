if '_base_':
    from ..retinanet.retinanet_r50_fpn_1x_coco import *
from mmdet.models.detectors.fsaf import FSAF
from mmdet.models.dense_heads.fsaf_head import FSAFHead
from mmdet.models.task_modules.prior_generators.anchor_generator import AnchorGenerator
from mmdet.models.task_modules.coders.tblr_bbox_coder import TBLRBBoxCoder
from mmdet.models.losses.focal_loss import FocalLoss
from mmdet.models.losses.iou_loss import IoULoss
from mmdet.models.task_modules.assigners.center_region_assigner import CenterRegionAssigner
# model settings
model.merge(
    dict(
        type=FSAF,
        bbox_head=dict(
            type=FSAFHead,
            num_classes=80,
            in_channels=256,
            stacked_convs=4,
            feat_channels=256,
            reg_decoded_bbox=True,
            # Only anchor-free branch is implemented. The anchor generator only
            #  generates 1 anchor at each feature point, as a substitute of the
            #  grid of features.
            anchor_generator=dict(
                type=AnchorGenerator,
                octave_base_scale=1,
                scales_per_octave=1,
                ratios=[1.0],
                strides=[8, 16, 32, 64, 128]),
            bbox_coder=dict(_delete_=True, type=TBLRBBoxCoder, normalizer=4.0),
            loss_cls=dict(
                type=FocalLoss,
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0,
                reduction='none'),
            loss_bbox=dict(
                _delete_=True,
                type=IoULoss,
                eps=1e-6,
                loss_weight=1.0,
                reduction='none')),
        # training and testing settings
        train_cfg=dict(
            assigner=dict(
                _delete_=True,
                type=CenterRegionAssigner,
                pos_scale=0.2,
                neg_scale=0.2,
                min_pos_iof=0.01),
            allowed_border=-1,
            pos_weight=-1,
            debug=False)))

optim_wrapper.merge(dict(clip_grad=dict(max_norm=10, norm_type=2)))
