# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
from mmengine import Config
from mmengine.structures import InstanceData

from mmdet import *  # noqa
from mmdet.models.dense_heads import VFNetHead


class TestVFNetHead(TestCase):

    def test_vfnet_head_loss(self):
        """Tests vfnet head loss when truth is empty and non-empty."""
        s = 256
        img_metas = [{
            'img_shape': (s, s, 3),
            'scale_factor': 1,
            'pad_shape': (s, s, 3)
        }]
        train_cfg = Config(
            dict(
                assigner=dict(type='ATSSAssigner', topk=9),
                allowed_border=-1,
                pos_weight=-1,
                debug=False))
        # since VarFocal Loss is not supported on CPU
        vfnet_head = VFNetHead(
            num_classes=4,
            in_channels=1,
            train_cfg=train_cfg,
            loss_cls=dict(
                type='VarifocalLoss', use_sigmoid=True, loss_weight=1.0))

        feat = [
            torch.rand(1, 1, s // feat_size, s // feat_size)
            for feat_size in [4, 8, 16, 32, 64]
        ]
        cls_scores, bbox_preds, bbox_preds_refine = vfnet_head.forward(feat)
        # Test that empty ground truth encourages the network to predict
        # background
        gt_instances = InstanceData()
        gt_instances.bboxes = torch.empty((0, 4))
        gt_instances.labels = torch.LongTensor([])
        empty_gt_losses = vfnet_head.loss_by_feat(cls_scores, bbox_preds,
                                                  bbox_preds_refine,
                                                  [gt_instances], img_metas)
        # When there is no truth, the cls loss should be nonzero but there
        # should be no box loss.
        empty_cls_loss = empty_gt_losses['loss_cls']
        empty_box_loss = empty_gt_losses['loss_bbox']
        self.assertGreater(empty_cls_loss.item(), 0,
                           'cls loss should be non-zero')
        self.assertEqual(
            empty_box_loss.item(), 0,
            'there should be no box loss when there are no true boxes')

        # When truth is non-empty then both cls and box loss should be nonzero
        # for random inputs
        gt_instances = InstanceData()
        gt_instances.bboxes = torch.Tensor(
            [[23.6667, 23.8757, 238.6326, 151.8874]])
        gt_instances.labels = torch.LongTensor([2])
        one_gt_losses = vfnet_head.loss_by_feat(cls_scores, bbox_preds,
                                                bbox_preds_refine,
                                                [gt_instances], img_metas)
        onegt_cls_loss = one_gt_losses['loss_cls']
        onegt_box_loss = one_gt_losses['loss_bbox']
        self.assertGreater(onegt_cls_loss.item(), 0,
                           'cls loss should be non-zero')
        self.assertGreater(onegt_box_loss.item(), 0,
                           'box loss should be non-zero')

    def test_vfnet_head_loss_without_atss(self):
        """Tests vfnet head loss when truth is empty and non-empty."""
        s = 256
        img_metas = [{
            'img_shape': (s, s, 3),
            'scale_factor': 1,
            'pad_shape': (s, s, 3)
        }]
        train_cfg = Config(
            dict(
                assigner=dict(type='ATSSAssigner', topk=9),
                allowed_border=-1,
                pos_weight=-1,
                debug=False))
        # since VarFocal Loss is not supported on CPU
        vfnet_head = VFNetHead(
            num_classes=4,
            in_channels=1,
            train_cfg=train_cfg,
            use_atss=False,
            loss_cls=dict(
                type='VarifocalLoss', use_sigmoid=True, loss_weight=1.0))

        feat = [
            torch.rand(1, 1, s // feat_size, s // feat_size)
            for feat_size in [4, 8, 16, 32, 64]
        ]
        cls_scores, bbox_preds, bbox_preds_refine = vfnet_head.forward(feat)
        # Test that empty ground truth encourages the network to predict
        # background
        gt_instances = InstanceData()
        gt_instances.bboxes = torch.empty((0, 4))
        gt_instances.labels = torch.LongTensor([])
        empty_gt_losses = vfnet_head.loss_by_feat(cls_scores, bbox_preds,
                                                  bbox_preds_refine,
                                                  [gt_instances], img_metas)
        # When there is no truth, the cls loss should be nonzero but there
        # should be no box loss.
        empty_cls_loss = empty_gt_losses['loss_cls']
        empty_box_loss = empty_gt_losses['loss_bbox']
        self.assertGreater(empty_cls_loss.item(), 0,
                           'cls loss should be non-zero')
        self.assertEqual(
            empty_box_loss.item(), 0,
            'there should be no box loss when there are no true boxes')

        # When truth is non-empty then both cls and box loss should be nonzero
        # for random inputs
        gt_instances = InstanceData()
        gt_instances.bboxes = torch.Tensor(
            [[23.6667, 23.8757, 238.6326, 151.8874]])
        gt_instances.labels = torch.LongTensor([2])
        one_gt_losses = vfnet_head.loss_by_feat(cls_scores, bbox_preds,
                                                bbox_preds_refine,
                                                [gt_instances], img_metas)
        onegt_cls_loss = one_gt_losses['loss_cls']
        onegt_box_loss = one_gt_losses['loss_bbox']
        self.assertGreater(onegt_cls_loss.item(), 0,
                           'cls loss should be non-zero')
        self.assertGreater(onegt_box_loss.item(), 0,
                           'box loss should be non-zero')
