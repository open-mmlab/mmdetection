if '_base_':
    from ..rpn.rpn_r50_caffe_fpn_1x_coco import *
from mmdet.models.dense_heads.cascade_rpn_head import CascadeRPNHead, StageCascadeRPNHead, StageCascadeRPNHead
from mmdet.models.task_modules.prior_generators.anchor_generator import AnchorGenerator
from mmdet.models.task_modules.coders.delta_xywh_bbox_coder import DeltaXYWHBBoxCoder, DeltaXYWHBBoxCoder
from mmdet.models.losses.iou_loss import IoULoss, IoULoss
from mmdet.models.losses.cross_entropy_loss import CrossEntropyLoss
from mmdet.models.task_modules.assigners.region_assigner import RegionAssigner
from mmdet.models.task_modules.assigners.max_iou_assigner import MaxIoUAssigner
from mmdet.models.task_modules.assigners.iou2d_calculator import BboxOverlaps2D
from mmdet.models.task_modules.samplers.random_sampler import RandomSampler

model.merge(
    dict(
        rpn_head=dict(
            _delete_=True,
            type=CascadeRPNHead,
            num_stages=2,
            stages=[
                dict(
                    type=StageCascadeRPNHead,
                    in_channels=256,
                    feat_channels=256,
                    anchor_generator=dict(
                        type=AnchorGenerator,
                        scales=[8],
                        ratios=[1.0],
                        strides=[4, 8, 16, 32, 64]),
                    adapt_cfg=dict(type='dilation', dilation=3),
                    bridged_feature=True,
                    sampling=False,
                    with_cls=False,
                    reg_decoded_bbox=True,
                    bbox_coder=dict(
                        type=DeltaXYWHBBoxCoder,
                        target_means=(.0, .0, .0, .0),
                        target_stds=(0.1, 0.1, 0.5, 0.5)),
                    loss_bbox=dict(
                        type=IoULoss, linear=True, loss_weight=10.0)),
                dict(
                    type=StageCascadeRPNHead,
                    in_channels=256,
                    feat_channels=256,
                    adapt_cfg=dict(type='offset'),
                    bridged_feature=False,
                    sampling=True,
                    with_cls=True,
                    reg_decoded_bbox=True,
                    bbox_coder=dict(
                        type=DeltaXYWHBBoxCoder,
                        target_means=(.0, .0, .0, .0),
                        target_stds=(0.05, 0.05, 0.1, 0.1)),
                    loss_cls=dict(
                        type=CrossEntropyLoss,
                        use_sigmoid=True,
                        loss_weight=1.0),
                    loss_bbox=dict(
                        type=IoULoss, linear=True, loss_weight=10.0))
            ]),
        train_cfg=dict(rpn=[
            dict(
                assigner=dict(
                    type=RegionAssigner, center_ratio=0.2, ignore_ratio=0.5),
                allowed_border=-1,
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type=MaxIoUAssigner,
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.7,
                    min_pos_iou=0.3,
                    ignore_iof_thr=-1,
                    iou_calculator=dict(type=BboxOverlaps2D)),
                sampler=dict(
                    type=RandomSampler,
                    num=256,
                    pos_fraction=0.5,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=False),
                allowed_border=-1,
                pos_weight=-1,
                debug=False)
        ]),
        test_cfg=dict(
            rpn=dict(
                nms_pre=2000,
                max_per_img=2000,
                nms=dict(type='nms', iou_threshold=0.8),
                min_bbox_size=0))))
optim_wrapper.merge(dict(clip_grad=dict(max_norm=35, norm_type=2)))
