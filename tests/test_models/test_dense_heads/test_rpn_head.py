# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import pytest
import torch
from mmengine import Config
from mmengine.structures import InstanceData

from mmdet import *  # noqa
from mmdet.models.dense_heads import RPNHead


class TestRPNHead(TestCase):

    def test_init(self):
        """Test init rpn head."""
        rpn_head = RPNHead(num_classes=1, in_channels=1)
        self.assertTrue(rpn_head.rpn_conv)
        self.assertTrue(rpn_head.rpn_cls)
        self.assertTrue(rpn_head.rpn_reg)

        # rpn_head.num_convs > 1
        rpn_head = RPNHead(num_classes=1, in_channels=1, num_convs=2)
        self.assertTrue(rpn_head.rpn_conv)
        self.assertTrue(rpn_head.rpn_cls)
        self.assertTrue(rpn_head.rpn_reg)

    def test_rpn_head_loss(self):
        """Tests rpn head loss when truth is empty and non-empty."""
        s = 256
        img_metas = [{
            'img_shape': (s, s, 3),
            'pad_shape': (s, s, 3),
            'scale_factor': 1,
        }]

        cfg = Config(
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.3,
                    min_pos_iou=0.3,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=256,
                    pos_fraction=0.5,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=False),
                allowed_border=0,
                pos_weight=-1,
                debug=False))
        rpn_head = RPNHead(num_classes=1, in_channels=1, train_cfg=cfg)

        # Anchor head expects a multiple levels of features per image
        feats = (
            torch.rand(1, 1, s // (2**(i + 2)), s // (2**(i + 2)))
            for i in range(len(rpn_head.prior_generator.strides)))
        cls_scores, bbox_preds = rpn_head.forward(feats)

        # Test that empty ground truth encourages the network to
        # predict background
        gt_instances = InstanceData()
        gt_instances.bboxes = torch.empty((0, 4))
        gt_instances.labels = torch.LongTensor([])

        empty_gt_losses = rpn_head.loss_by_feat(cls_scores, bbox_preds,
                                                [gt_instances], img_metas)
        # When there is no truth, the cls loss should be nonzero but
        # there should be no box loss.
        empty_cls_loss = sum(empty_gt_losses['loss_rpn_cls'])
        empty_box_loss = sum(empty_gt_losses['loss_rpn_bbox'])
        self.assertGreater(empty_cls_loss.item(), 0,
                           'rpn cls loss should be non-zero')
        self.assertEqual(
            empty_box_loss.item(), 0,
            'there should be no box loss when there are no true boxes')

        # When truth is non-empty then both cls and box loss
        # should be nonzero for random inputs
        gt_instances = InstanceData()
        gt_instances.bboxes = torch.Tensor(
            [[23.6667, 23.8757, 238.6326, 151.8874]])
        gt_instances.labels = torch.LongTensor([0])

        one_gt_losses = rpn_head.loss_by_feat(cls_scores, bbox_preds,
                                              [gt_instances], img_metas)
        onegt_cls_loss = sum(one_gt_losses['loss_rpn_cls'])
        onegt_box_loss = sum(one_gt_losses['loss_rpn_bbox'])
        self.assertGreater(onegt_cls_loss.item(), 0,
                           'rpn cls loss should be non-zero')
        self.assertGreater(onegt_box_loss.item(), 0,
                           'rpn box loss should be non-zero')

        # When there is no valid anchor, the loss will be None,
        # and this will raise a ValueError.
        img_metas = [{
            'img_shape': (8, 8, 3),
            'pad_shape': (8, 8, 3),
            'scale_factor': 1,
        }]
        with pytest.raises(ValueError):
            rpn_head.loss_by_feat(cls_scores, bbox_preds, [gt_instances],
                                  img_metas)

    def test_bbox_post_process(self):
        """Test the length of detection instance results is 0."""
        from mmengine.config import ConfigDict
        cfg = ConfigDict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0)

        rpn_head = RPNHead(num_classes=1, in_channels=1)
        results = InstanceData(metainfo=dict())
        results.bboxes = torch.zeros((0, 4))
        results.scores = torch.zeros(0)
        results = rpn_head._bbox_post_process(results, cfg, img_meta=dict())
        self.assertEqual(len(results), 0)
        self.assertEqual(results.bboxes.size(), (0, 4))
        self.assertEqual(results.scores.size(), (0, ))
        self.assertEqual(results.labels.size(), (0, ))
