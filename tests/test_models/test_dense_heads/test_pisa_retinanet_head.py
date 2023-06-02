# Copyright (c) OpenMMLab. All rights reserved.
from math import ceil
from unittest import TestCase

import torch
from mmengine import Config
from mmengine.structures import InstanceData

from mmdet import *  # noqa
from mmdet.models.dense_heads import PISARetinaHead


class TestPISARetinaHead(TestCase):

    def test_pisa_reitnanet_head_loss(self):
        """Tests pisa retinanet head loss when truth is empty and non-empty."""
        s = 300
        img_metas = [{
            'img_shape': (s, s),
            'pad_shape': (s, s),
            'scale_factor': 1,
        }]
        cfg = Config(
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.4,
                    min_pos_iou=0,
                    ignore_iof_thr=-1),
                isr=dict(k=2., bias=0.),
                carl=dict(k=1., bias=0.2),
                sampler=dict(type='PseudoSampler'),
                allowed_border=-1,
                pos_weight=-1,
                debug=False))
        pisa_retinanet_head = PISARetinaHead(
            num_classes=4,
            in_channels=1,
            stacked_convs=1,
            feat_channels=256,
            anchor_generator=dict(
                type='AnchorGenerator',
                octave_base_scale=4,
                scales_per_octave=3,
                ratios=[0.5, 1.0, 2.0],
                strides=[8, 16, 32, 64, 128]),
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[.0, .0, .0, .0],
                target_stds=[1.0, 1.0, 1.0, 1.0]),
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=0.11, loss_weight=1.0),
            train_cfg=cfg)

        # pisa retina head expects a multiple levels of features per image
        feats = (
            torch.rand(1, 1, ceil(s / stride[0]), ceil(s / stride[0]))
            for stride in pisa_retinanet_head.prior_generator.strides)
        cls_scores, bbox_preds = pisa_retinanet_head.forward(feats)

        # Test that empty ground truth encourages the network to
        # predict background
        gt_instances = InstanceData()
        gt_instances.bboxes = torch.empty((0, 4))
        gt_instances.labels = torch.LongTensor([])

        empty_gt_losses = pisa_retinanet_head.loss_by_feat(
            cls_scores, bbox_preds, [gt_instances], img_metas)
        # When there is no truth, cls_loss and box_loss should all be zero.
        empty_cls_loss = empty_gt_losses['loss_cls']
        empty_box_loss = empty_gt_losses['loss_bbox']
        empty_carl_loss = empty_gt_losses['loss_carl']
        self.assertGreater(empty_cls_loss.item(), 0,
                           'cls loss should be non-zero')
        self.assertEqual(
            empty_box_loss.item(), 0,
            'there should be no box loss when there are no true boxes')
        self.assertEqual(
            empty_carl_loss.item(), 0,
            'there should be no carl loss when there are no true boxes')

        # When truth is non-empty then both cls and box loss
        # should be nonzero for random inputs
        gt_instances = InstanceData()
        gt_instances.bboxes = torch.Tensor(
            [[23.6667, 23.8757, 238.6326, 151.8874]])
        gt_instances.labels = torch.LongTensor([2])

        one_gt_losses = pisa_retinanet_head.loss_by_feat(
            cls_scores, bbox_preds, [gt_instances], img_metas)
        onegt_cls_loss = one_gt_losses['loss_cls']
        onegt_box_loss = one_gt_losses['loss_bbox']
        onegt_carl_loss = one_gt_losses['loss_carl']
        self.assertGreater(onegt_cls_loss.item(), 0,
                           'cls loss should be non-zero')
        self.assertGreater(onegt_box_loss.item(), 0,
                           'box loss should be non-zero')
        self.assertGreater(onegt_carl_loss.item(), 0,
                           'carl loss should be non-zero')
