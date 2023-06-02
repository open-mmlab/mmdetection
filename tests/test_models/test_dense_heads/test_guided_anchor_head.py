# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData

from mmdet.models.dense_heads import GuidedAnchorHead

guided_anchor_head_config = ConfigDict(
    dict(
        num_classes=4,
        in_channels=4,
        feat_channels=4,
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


class TestGuidedAnchorHead(TestCase):

    def test_guided_anchor_head_loss(self):
        """Tests guided anchor loss when truth is empty and non-empty."""
        s = 256
        img_metas = [{
            'img_shape': (s, s),
            'pad_shape': (s, s),
            'scale_factor': (1, 1)
        }]
        guided_anchor_head = GuidedAnchorHead(**guided_anchor_head_config)

        feats = (
            torch.rand(1, 4, s // stride[1], s // stride[0])
            for stride in guided_anchor_head.square_anchor_generator.strides)
        outs = guided_anchor_head(feats)

        # Test that empty ground truth encourages the network to
        # predict background
        gt_instances = InstanceData()
        gt_instances.bboxes = torch.empty((0, 4))
        gt_instances.labels = torch.LongTensor([])

        empty_gt_losses = guided_anchor_head.loss_by_feat(
            *outs, [gt_instances], img_metas)
        # When there is no truth, the cls loss should be nonzero but
        # box shape and location loss should be zero
        empty_cls_loss = sum(empty_gt_losses['loss_cls']).item()
        empty_box_loss = sum(empty_gt_losses['loss_bbox']).item()
        empty_shape_loss = sum(empty_gt_losses['loss_shape']).item()
        empty_loc_loss = sum(empty_gt_losses['loss_loc']).item()
        self.assertGreater(empty_cls_loss, 0, 'cls loss should be non-zero')
        self.assertGreater(empty_loc_loss, 0,
                           'location loss should be non-zero')
        self.assertEqual(
            empty_box_loss, 0,
            'there should be no box loss when there are no true boxes')
        self.assertEqual(
            empty_shape_loss, 0,
            'there should be no shape loss when there are no true boxes')

        # When truth is non-empty then all cls, box loss and centerness loss
        # should be nonzero for random inputs
        gt_instances = InstanceData()
        gt_instances.bboxes = torch.Tensor(
            [[23.6667, 23.8757, 238.6326, 151.8874]])
        gt_instances.labels = torch.LongTensor([2])

        one_gt_losses = guided_anchor_head.loss_by_feat(
            *outs, [gt_instances], img_metas)
        onegt_cls_loss = sum(one_gt_losses['loss_cls']).item()
        onegt_box_loss = sum(one_gt_losses['loss_bbox']).item()
        onegt_shape_loss = sum(one_gt_losses['loss_shape']).item()
        onegt_loc_loss = sum(one_gt_losses['loss_loc']).item()
        self.assertGreater(onegt_cls_loss, 0, 'cls loss should be non-zero')
        self.assertGreater(onegt_box_loss, 0, 'box loss should be non-zero')
        self.assertGreater(onegt_shape_loss, 0,
                           'shape loss should be non-zero')
        self.assertGreater(onegt_loc_loss, 0,
                           'location loss should be non-zero')

    def test_guided_anchor_head_predict_by_feat(self):
        s = 256
        img_metas = [{
            'img_shape': (s, s),
            'pad_shape': (s, s),
            'scale_factor': (1, 1)
        }]
        guided_anchor_head = GuidedAnchorHead(**guided_anchor_head_config)

        feats = (
            torch.rand(1, 4, s // stride[1], s // stride[0])
            for stride in guided_anchor_head.square_anchor_generator.strides)
        outs = guided_anchor_head(feats)

        guided_anchor_head.predict_by_feat(
            *outs, batch_img_metas=img_metas, rescale=True)
