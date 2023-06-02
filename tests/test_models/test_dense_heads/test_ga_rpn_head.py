# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData

from mmdet.models.dense_heads import GARPNHead

ga_rpn_config = ConfigDict(
    dict(
        num_classes=1,
        in_channels=4,
        feat_channels=4,
        approx_anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=8,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        square_anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            scales=[8],
            strides=[4, 8, 16, 32, 64]),
        anchor_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.07, 0.07, 0.14, 0.14]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.07, 0.07, 0.11, 0.11]),
        loc_filter_thr=0.01,
        loss_loc=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_shape=dict(type='BoundedIoULoss', beta=0.2, loss_weight=1.0),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0),
        train_cfg=dict(
            ga_assigner=dict(
                type='ApproxMaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                ignore_iof_thr=-1),
            ga_sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            center_ratio=0.2,
            ignore_ratio=0.5,
            pos_weight=-1,
            debug=False),
        test_cfg=dict(
            nms_pre=1000,
            ms_post=1000,
            max_per_img=300,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0)))


class TestGARPNHead(TestCase):

    def test_ga_rpn_head_loss(self):
        """Tests ga rpn head loss."""
        s = 256
        img_metas = [{
            'img_shape': (s, s),
            'pad_shape': (s, s),
            'scale_factor': (1, 1)
        }]
        ga_rpn_head = GARPNHead(**ga_rpn_config)

        feats = (
            torch.rand(1, 4, s // stride[1], s // stride[0])
            for stride in ga_rpn_head.square_anchor_generator.strides)
        outs = ga_rpn_head(feats)

        # When truth is non-empty then all cls, box loss and centerness loss
        # should be nonzero for random inputs
        gt_instances = InstanceData()
        gt_instances.bboxes = torch.Tensor(
            [[23.6667, 23.8757, 238.6326, 151.8874]])
        gt_instances.labels = torch.LongTensor([0])

        one_gt_losses = ga_rpn_head.loss_by_feat(*outs, [gt_instances],
                                                 img_metas)
        onegt_cls_loss = sum(one_gt_losses['loss_rpn_cls']).item()
        onegt_box_loss = sum(one_gt_losses['loss_rpn_bbox']).item()
        onegt_shape_loss = sum(one_gt_losses['loss_anchor_shape']).item()
        onegt_loc_loss = sum(one_gt_losses['loss_anchor_loc']).item()
        self.assertGreater(onegt_cls_loss, 0, 'cls loss should be non-zero')
        self.assertGreater(onegt_box_loss, 0, 'box loss should be non-zero')
        self.assertGreater(onegt_shape_loss, 0,
                           'shape loss should be non-zero')
        self.assertGreater(onegt_loc_loss, 0,
                           'location loss should be non-zero')

    def test_ga_rpn_head_predict_by_feat(self):
        s = 256
        img_metas = [{
            'img_shape': (s, s),
            'pad_shape': (s, s),
            'scale_factor': (1, 1)
        }]
        ga_rpn_head = GARPNHead(**ga_rpn_config)

        feats = (
            torch.rand(1, 4, s // stride[1], s // stride[0])
            for stride in ga_rpn_head.square_anchor_generator.strides)
        outs = ga_rpn_head(feats)

        cfg = ConfigDict(
            dict(
                nms_pre=2000,
                nms_post=1000,
                max_per_img=300,
                nms=dict(type='nms', iou_threshold=0.7),
                min_bbox_size=0))
        ga_rpn_head.predict_by_feat(
            *outs, batch_img_metas=img_metas, cfg=cfg, rescale=True)
