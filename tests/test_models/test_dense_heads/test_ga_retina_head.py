# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
from mmengine.config import ConfigDict

from mmdet.models.dense_heads import GARetinaHead

ga_retina_head_config = ConfigDict(
    dict(
        num_classes=4,
        in_channels=4,
        feat_channels=4,
        stacked_convs=1,
        approx_anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128]),
        square_anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            scales=[4],
            strides=[8, 16, 32, 64, 128]),
        anchor_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loc_filter_thr=0.01,
        loss_loc=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_shape=dict(type='BoundedIoULoss', beta=0.2, loss_weight=1.0),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=0.04, loss_weight=1.0),
        train_cfg=dict(
            ga_assigner=dict(
                type='ApproxMaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.4,
                min_pos_iou=0.4,
                ignore_iof_thr=-1),
            ga_sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.0,
                ignore_iof_thr=-1),
            allowed_border=-1,
            pos_weight=-1,
            center_ratio=0.2,
            ignore_ratio=0.5,
            debug=False),
        test_cfg=dict(
            nms_pre=1000,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100)))


class TestGARetinaHead(TestCase):

    def test_ga_retina_head_init_and_forward(self):
        """The GARetinaHead inherit loss and prediction function from
        GuidedAchorHead.

        Here, we only test GARetinaHet initialization and forward.
        """
        # Test initializaion
        ga_retina_head = GARetinaHead(**ga_retina_head_config)

        # Test forward
        s = 256
        feats = (
            torch.rand(1, 4, s // stride[1], s // stride[0])
            for stride in ga_retina_head.square_anchor_generator.strides)
        ga_retina_head(feats)
