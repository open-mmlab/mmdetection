# Copyright (c) OpenMMLab. All rights reserved.
from math import ceil
from unittest import TestCase

import torch
from mmengine import Config
from mmengine.structures import InstanceData

from mmdet.models.dense_heads import FSAFHead


class TestFSAFHead(TestCase):

    def test_fsaf_head_loss(self):
        """Tests fsaf head loss when truth is empty and non-empty."""
        s = 300
        img_metas = [{
            'img_shape': (s, s),
            'pad_shape': (s, s),
            'scale_factor': 1,
        }]
        cfg = Config(
            dict(
                assigner=dict(
                    type='CenterRegionAssigner',
                    pos_scale=0.2,
                    neg_scale=0.2,
                    min_pos_iof=0.01),
                allowed_border=-1,
                pos_weight=-1,
                debug=False))
        fsaf_head = FSAFHead(
            num_classes=4,
            in_channels=1,
            stacked_convs=1,
            feat_channels=1,
            reg_decoded_bbox=True,
            anchor_generator=dict(
                type='AnchorGenerator',
                octave_base_scale=1,
                scales_per_octave=1,
                ratios=[1.0],
                strides=[8, 16, 32, 64, 128]),
            bbox_coder=dict(type='TBLRBBoxCoder', normalizer=4.0),
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0,
                reduction='none'),
            loss_bbox=dict(
                type='IoULoss', eps=1e-6, loss_weight=1.0, reduction='none'),
            train_cfg=cfg)

        # FSAF head expects a multiple levels of features per image
        feats = (
            torch.rand(1, 1, ceil(s / stride[0]), ceil(s / stride[0]))
            for stride in fsaf_head.prior_generator.strides)
        cls_scores, bbox_preds = fsaf_head.forward(feats)

        # Test that empty ground truth encourages the network to
        # predict background
        gt_instances = InstanceData()
        gt_instances.bboxes = torch.empty((0, 4))
        gt_instances.labels = torch.LongTensor([])

        empty_gt_losses = fsaf_head.loss_by_feat(cls_scores, bbox_preds,
                                                 [gt_instances], img_metas)
        # When there is no truth, the cls loss should be nonzero but
        # box loss should be zero
        empty_cls_loss = sum(empty_gt_losses['loss_cls'])
        empty_box_loss = sum(empty_gt_losses['loss_bbox'])
        self.assertGreater(empty_cls_loss, 0, 'cls loss should be non-zero')
        self.assertEqual(
            empty_box_loss.item(), 0,
            'there should be no box loss when there are no true boxes')

        # When truth is non-empty then both cls and box loss
        # should be nonzero for random inputs
        gt_instances = InstanceData()
        gt_instances.bboxes = torch.Tensor(
            [[23.6667, 23.8757, 238.6326, 151.8874]])
        gt_instances.labels = torch.LongTensor([2])

        one_gt_losses = fsaf_head.loss_by_feat(cls_scores, bbox_preds,
                                               [gt_instances], img_metas)
        onegt_cls_loss = sum(one_gt_losses['loss_cls'])
        onegt_box_loss = sum(one_gt_losses['loss_bbox'])
        self.assertGreater(onegt_cls_loss.item(), 0,
                           'cls loss should be non-zero')
        self.assertGreater(onegt_box_loss.item(), 0,
                           'box loss should be non-zero')
