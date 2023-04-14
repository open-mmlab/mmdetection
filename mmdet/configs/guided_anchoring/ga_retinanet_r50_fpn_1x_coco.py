if '_base_':
    from ..retinanet.retinanet_r50_fpn_1x_coco import *
from mmdet.models.dense_heads.ga_retina_head import GARetinaHead
from mmdet.models.task_modules.prior_generators.anchor_generator import AnchorGenerator, AnchorGenerator
from mmdet.models.task_modules.coders.delta_xywh_bbox_coder import DeltaXYWHBBoxCoder, DeltaXYWHBBoxCoder
from mmdet.models.losses.focal_loss import FocalLoss, FocalLoss
from mmdet.models.losses.iou_loss import BoundedIoULoss
from mmdet.models.losses.smooth_l1_loss import SmoothL1Loss
from mmdet.models.task_modules.assigners.approx_max_iou_assigner import ApproxMaxIoUAssigner
from mmdet.models.task_modules.samplers.random_sampler import RandomSampler

model.merge(
    dict(
        bbox_head=dict(
            _delete_=True,
            type=GARetinaHead,
            num_classes=80,
            in_channels=256,
            stacked_convs=4,
            feat_channels=256,
            approx_anchor_generator=dict(
                type=AnchorGenerator,
                octave_base_scale=4,
                scales_per_octave=3,
                ratios=[0.5, 1.0, 2.0],
                strides=[8, 16, 32, 64, 128]),
            square_anchor_generator=dict(
                type=AnchorGenerator,
                ratios=[1.0],
                scales=[4],
                strides=[8, 16, 32, 64, 128]),
            anchor_coder=dict(
                type=DeltaXYWHBBoxCoder,
                target_means=[.0, .0, .0, .0],
                target_stds=[1.0, 1.0, 1.0, 1.0]),
            bbox_coder=dict(
                type=DeltaXYWHBBoxCoder,
                target_means=[.0, .0, .0, .0],
                target_stds=[1.0, 1.0, 1.0, 1.0]),
            loc_filter_thr=0.01,
            loss_loc=dict(
                type=FocalLoss,
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0),
            loss_shape=dict(type=BoundedIoULoss, beta=0.2, loss_weight=1.0),
            loss_cls=dict(
                type=FocalLoss,
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0),
            loss_bbox=dict(type=SmoothL1Loss, beta=0.04, loss_weight=1.0)),
        # training and testing settings
        train_cfg=dict(
            ga_assigner=dict(
                type=ApproxMaxIoUAssigner,
                pos_iou_thr=0.5,
                neg_iou_thr=0.4,
                min_pos_iou=0.4,
                ignore_iof_thr=-1),
            ga_sampler=dict(
                type=RandomSampler,
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            assigner=dict(neg_iou_thr=0.5, min_pos_iou=0.0),
            center_ratio=0.2,
            ignore_ratio=0.5)))
optim_wrapper.merge(dict(clip_grad=dict(max_norm=35, norm_type=2)))
