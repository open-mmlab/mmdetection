# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
from mmengine import Config
from mmengine.structures import InstanceData

from mmdet import *  # noqa
from mmdet.models.dense_heads import DDODHead


class TestDDODHead(TestCase):

    def test_ddod_head_loss(self):
        """Tests ddod head loss when truth is empty and non-empty."""
        s = 256
        img_metas = [{
            'img_shape': (s, s, 3),
            'pad_shape': (s, s, 3),
            'scale_factor': 1
        }]
        cfg = Config(
            dict(
                assigner=dict(type='ATSSAssigner', topk=9, alpha=0.8),
                reg_assigner=dict(type='ATSSAssigner', topk=9, alpha=0.5),
                allowed_border=-1,
                pos_weight=-1,
                debug=False))
        atss_head = DDODHead(
            num_classes=4,
            in_channels=1,
            stacked_convs=1,
            feat_channels=1,
            use_dcn=False,
            norm_cfg=None,
            train_cfg=cfg,
            anchor_generator=dict(
                type='AnchorGenerator',
                ratios=[1.0],
                octave_base_scale=8,
                scales_per_octave=1,
                strides=[8, 16, 32, 64, 128]),
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0),
            loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
            loss_iou=dict(
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0))
        feat = [
            torch.rand(1, 1, s // feat_size, s // feat_size)
            for feat_size in [8, 16, 32, 64, 128]
        ]
        cls_scores, bbox_preds, centernesses = atss_head.forward(feat)

        # Test that empty ground truth encourages the network to predict
        # background
        gt_instances = InstanceData()
        gt_instances.bboxes = torch.empty((0, 4))
        gt_instances.labels = torch.LongTensor([])

        empty_gt_losses = atss_head.loss_by_feat(cls_scores, bbox_preds,
                                                 centernesses, [gt_instances],
                                                 img_metas)
        # When there is no truth, the cls loss should be nonzero but there
        # should be no box loss.
        empty_cls_loss = sum(empty_gt_losses['loss_cls'])
        empty_box_loss = sum(empty_gt_losses['loss_bbox'])
        empty_centerness_loss = sum(empty_gt_losses['loss_iou'])
        self.assertGreater(empty_cls_loss.item(), 0,
                           'cls loss should be non-zero')
        self.assertEqual(
            empty_box_loss.item(), 0,
            'there should be no box loss when there are no true boxes')
        self.assertEqual(
            empty_centerness_loss.item(), 0,
            'there should be no centerness loss when there are no true boxes')

        # When truth is non-empty then both cls and box loss should be nonzero
        # for random inputs
        gt_instances = InstanceData()
        gt_instances.bboxes = torch.Tensor(
            [[23.6667, 23.8757, 238.6326, 151.8874]])
        gt_instances.labels = torch.LongTensor([2])
        one_gt_losses = atss_head.loss_by_feat(cls_scores, bbox_preds,
                                               centernesses, [gt_instances],
                                               img_metas)
        onegt_cls_loss = sum(one_gt_losses['loss_cls'])
        onegt_box_loss = sum(one_gt_losses['loss_bbox'])
        onegt_centerness_loss = sum(one_gt_losses['loss_iou'])
        self.assertGreater(onegt_cls_loss.item(), 0,
                           'cls loss should be non-zero')
        self.assertGreater(onegt_box_loss.item(), 0,
                           'box loss should be non-zero')
        self.assertGreater(onegt_centerness_loss.item(), 0,
                           'centerness loss should be non-zero')
