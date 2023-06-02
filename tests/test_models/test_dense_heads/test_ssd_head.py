# Copyright (c) OpenMMLab. All rights reserved.
from math import ceil
from unittest import TestCase

import torch
from mmengine import Config
from mmengine.structures import InstanceData

from mmdet import *  # noqa
from mmdet.models.dense_heads import SSDHead


class TestSSDHead(TestCase):

    def test_ssd_head_loss(self):
        """Tests ssd head loss when truth is empty and non-empty."""
        s = 300
        img_metas = [{
            'img_shape': (s, s, 3),
            'pad_shape': (s, s, 3),
            'scale_factor': 1,
        }]
        cfg = Config(
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.,
                    ignore_iof_thr=-1,
                    gt_max_assign_all=False),
                sampler=dict(type='PseudoSampler'),
                smoothl1_beta=1.,
                allowed_border=-1,
                pos_weight=-1,
                neg_pos_ratio=3,
                debug=False))
        ssd_head = SSDHead(
            num_classes=4,
            in_channels=(1, 1, 1, 1, 1, 1),
            stacked_convs=1,
            feat_channels=1,
            use_depthwise=True,
            anchor_generator=dict(
                type='SSDAnchorGenerator',
                scale_major=False,
                input_size=s,
                basesize_ratio_range=(0.15, 0.9),
                strides=[8, 16, 32, 64, 100, 300],
                ratios=[[2], [2, 3], [2, 3], [2, 3], [2], [2]]),
            train_cfg=cfg)

        # SSD head expects a multiple levels of features per image
        feats = (
            torch.rand(1, 1, ceil(s / stride[0]), ceil(s / stride[0]))
            for stride in ssd_head.prior_generator.strides)
        cls_scores, bbox_preds = ssd_head.forward(feats)

        # Test that empty ground truth encourages the network to
        # predict background
        gt_instances = InstanceData()
        gt_instances.bboxes = torch.empty((0, 4))
        gt_instances.labels = torch.LongTensor([])

        empty_gt_losses = ssd_head.loss_by_feat(cls_scores, bbox_preds,
                                                [gt_instances], img_metas)
        # When there is no truth, cls_loss and box_loss should all be zero.
        empty_cls_loss = sum(empty_gt_losses['loss_cls'])
        empty_box_loss = sum(empty_gt_losses['loss_bbox'])
        self.assertEqual(
            empty_cls_loss.item(), 0,
            'there should be no cls loss when there are no true boxes')
        self.assertEqual(
            empty_box_loss.item(), 0,
            'there should be no box loss when there are no true boxes')

        # When truth is non-empty then both cls and box loss
        # should be nonzero for random inputs
        gt_instances = InstanceData()
        gt_instances.bboxes = torch.Tensor(
            [[23.6667, 23.8757, 238.6326, 151.8874]])
        gt_instances.labels = torch.LongTensor([2])

        one_gt_losses = ssd_head.loss_by_feat(cls_scores, bbox_preds,
                                              [gt_instances], img_metas)
        onegt_cls_loss = sum(one_gt_losses['loss_cls'])
        onegt_box_loss = sum(one_gt_losses['loss_bbox'])
        self.assertGreater(onegt_cls_loss.item(), 0,
                           'cls loss should be non-zero')
        self.assertGreater(onegt_box_loss.item(), 0,
                           'box loss should be non-zero')
