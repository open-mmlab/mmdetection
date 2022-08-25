# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
from mmengine import Config, MessageHub
from mmengine.structures import InstanceData

from mmdet import *  # noqa
from mmdet.models.dense_heads import TOODHead


def _tood_head(anchor_type):
    """Set type of tood head."""
    train_cfg = Config(
        dict(
            initial_epoch=4,
            initial_assigner=dict(type='ATSSAssigner', topk=9),
            assigner=dict(type='TaskAlignedAssigner', topk=13),
            alpha=1,
            beta=6,
            allowed_border=-1,
            pos_weight=-1,
            debug=False))
    test_cfg = Config(
        dict(
            nms_pre=1000,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.6),
            max_per_img=100))

    tood_head = TOODHead(
        num_classes=80,
        in_channels=1,
        stacked_convs=1,
        feat_channels=8,  # the same as `la_down_rate` in TaskDecomposition
        norm_cfg=None,
        anchor_type=anchor_type,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        initial_loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            activated=True,  # use probability instead of logit as input
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            activated=True,  # use probability instead of logit as input
            beta=2.0,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        train_cfg=train_cfg,
        test_cfg=test_cfg)
    return tood_head


class TestTOODHead(TestCase):

    def test_tood_head_anchor_free_loss(self):
        """Tests tood head loss when truth is empty and non-empty."""
        s = 256
        img_metas = [{
            'img_shape': (s, s, 3),
            'pad_shape': (s, s, 3),
            'scale_factor': 1
        }]
        tood_head = _tood_head('anchor_free')
        tood_head.init_weights()
        feat = [
            torch.rand(1, 1, s // feat_size, s // feat_size)
            for feat_size in [8, 16, 32, 64, 128]
        ]
        cls_scores, bbox_preds = tood_head(feat)

        message_hub = MessageHub.get_instance('runtime_info')
        message_hub.update_info('epoch', 0)
        # Test that empty ground truth encourages the network to predict
        # background
        gt_instances = InstanceData()
        gt_instances.bboxes = torch.empty((0, 4))
        gt_instances.labels = torch.LongTensor([])
        gt_bboxes_ignore = None
        empty_gt_losses = tood_head.loss_by_feat(cls_scores, bbox_preds,
                                                 [gt_instances], img_metas,
                                                 gt_bboxes_ignore)
        # When there is no truth, the cls loss should be nonzero but there
        # should be no box loss.
        empty_cls_loss = empty_gt_losses['loss_cls']
        empty_box_loss = empty_gt_losses['loss_bbox']
        self.assertGreater(
            sum(empty_cls_loss).item(), 0, 'cls loss should be non-zero')
        self.assertEqual(
            sum(empty_box_loss).item(), 0,
            'there should be no box loss when there are no true boxes')
        # When truth is non-empty then both cls and box loss should be nonzero
        # for random inputs
        gt_instances = InstanceData()
        gt_instances.bboxes = torch.Tensor(
            [[23.6667, 23.8757, 238.6326, 151.8874]])
        gt_instances.labels = torch.LongTensor([2])
        gt_bboxes_ignore = None
        one_gt_losses = tood_head.loss_by_feat(cls_scores, bbox_preds,
                                               [gt_instances], img_metas,
                                               gt_bboxes_ignore)
        onegt_cls_loss = one_gt_losses['loss_cls']
        onegt_box_loss = one_gt_losses['loss_bbox']
        self.assertGreater(
            sum(onegt_cls_loss).item(), 0, 'cls loss should be non-zero')
        self.assertGreater(
            sum(onegt_box_loss).item(), 0, 'box loss should be non-zero')

        # Test that empty ground truth encourages the network to predict
        # background
        gt_instances = InstanceData()
        gt_instances.bboxes = torch.empty((0, 4))
        gt_instances.labels = torch.LongTensor([])
        gt_bboxes_ignore = None
        empty_gt_losses = tood_head.loss_by_feat(cls_scores, bbox_preds,
                                                 [gt_instances], img_metas,
                                                 gt_bboxes_ignore)
        # When there is no truth, the cls loss should be nonzero but there
        # should be no box loss.
        empty_cls_loss = empty_gt_losses['loss_cls']
        empty_box_loss = empty_gt_losses['loss_bbox']
        self.assertGreater(
            sum(empty_cls_loss).item(), 0, 'cls loss should be non-zero')
        self.assertEqual(
            sum(empty_box_loss).item(), 0,
            'there should be no box loss when there are no true boxes')
        # When truth is non-empty then both cls and box loss should be nonzero
        # for random inputs
        gt_instances = InstanceData()
        gt_instances.bboxes = torch.Tensor(
            [[23.6667, 23.8757, 238.6326, 151.8874]])
        gt_instances.labels = torch.LongTensor([2])
        gt_bboxes_ignore = None
        one_gt_losses = tood_head.loss_by_feat(cls_scores, bbox_preds,
                                               [gt_instances], img_metas,
                                               gt_bboxes_ignore)
        onegt_cls_loss = one_gt_losses['loss_cls']
        onegt_box_loss = one_gt_losses['loss_bbox']
        self.assertGreater(
            sum(onegt_cls_loss).item(), 0, 'cls loss should be non-zero')
        self.assertGreater(
            sum(onegt_box_loss).item(), 0, 'box loss should be non-zero')

    def test_tood_head_anchor_based_loss(self):
        """Tests tood head loss when truth is empty and non-empty."""
        s = 256
        img_metas = [{
            'img_shape': (s, s, 3),
            'pad_shape': (s, s, 3),
            'scale_factor': 1
        }]
        tood_head = _tood_head('anchor_based')
        tood_head.init_weights()
        feat = [
            torch.rand(1, 1, s // feat_size, s // feat_size)
            for feat_size in [8, 16, 32, 64, 128]
        ]
        cls_scores, bbox_preds = tood_head(feat)

        message_hub = MessageHub.get_instance('runtime_info')
        message_hub.update_info('epoch', 0)
        # Test that empty ground truth encourages the network to predict
        # background
        gt_instances = InstanceData()
        gt_instances.bboxes = torch.empty((0, 4))
        gt_instances.labels = torch.LongTensor([])
        gt_bboxes_ignore = None
        empty_gt_losses = tood_head.loss_by_feat(cls_scores, bbox_preds,
                                                 [gt_instances], img_metas,
                                                 gt_bboxes_ignore)
        # When there is no truth, the cls loss should be nonzero but there
        # should be no box loss.
        empty_cls_loss = empty_gt_losses['loss_cls']
        empty_box_loss = empty_gt_losses['loss_bbox']
        self.assertGreater(
            sum(empty_cls_loss).item(), 0, 'cls loss should be non-zero')
        self.assertEqual(
            sum(empty_box_loss).item(), 0,
            'there should be no box loss when there are no true boxes')
