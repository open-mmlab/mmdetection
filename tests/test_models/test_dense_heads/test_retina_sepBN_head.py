# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
from mmengine import Config
from mmengine.structures import InstanceData

from mmdet import *  # noqa
from mmdet.models.dense_heads import RetinaSepBNHead


class TestRetinaSepBNHead(TestCase):

    def test_init(self):
        """Test init RetinaSepBN head."""
        anchor_head = RetinaSepBNHead(num_classes=1, num_ins=1, in_channels=1)
        anchor_head.init_weights()
        self.assertTrue(anchor_head.cls_convs)
        self.assertTrue(anchor_head.reg_convs)
        self.assertTrue(anchor_head.retina_cls)
        self.assertTrue(anchor_head.retina_reg)

    def test_retina_sepbn_head_loss(self):
        """Tests RetinaSepBN head loss when truth is empty and non-empty."""
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
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.4,
                    min_pos_iou=0,
                    ignore_iof_thr=-1),
                sampler=dict(type='PseudoSampler'
                             ),  # Focal loss should use PseudoSampler
                allowed_border=-1,
                pos_weight=-1,
                debug=False))
        anchor_head = RetinaSepBNHead(
            num_classes=4, num_ins=5, in_channels=1, train_cfg=cfg)

        # Anchor head expects a multiple levels of features per image
        feats = []
        for i in range(len(anchor_head.prior_generator.strides)):
            feats.append(
                torch.rand(1, 1, s // (2**(i + 2)), s // (2**(i + 2))))

        cls_scores, bbox_preds = anchor_head.forward(tuple(feats))

        # Test that empty ground truth encourages the network to
        # predict background
        gt_instances = InstanceData()
        gt_instances.bboxes = torch.empty((0, 4))
        gt_instances.labels = torch.LongTensor([])

        empty_gt_losses = anchor_head.loss_by_feat(cls_scores, bbox_preds,
                                                   [gt_instances], img_metas)
        # When there is no truth, the cls loss should be nonzero but
        # there should be no box loss.
        empty_cls_loss = sum(empty_gt_losses['loss_cls'])
        empty_box_loss = sum(empty_gt_losses['loss_bbox'])
        self.assertGreater(empty_cls_loss.item(), 0,
                           'cls loss should be non-zero')
        self.assertEqual(
            empty_box_loss.item(), 0,
            'there should be no box loss when there are no true boxes')

        # When truth is non-empty then both cls and box loss
        # should be nonzero for random inputs
        gt_instances = InstanceData()
        gt_instances.bboxes = torch.Tensor(
            [[23.6667, 23.8757, 238.6326, 151.8874]])
        gt_instances.labels = torch.LongTensor([2])

        one_gt_losses = anchor_head.loss_by_feat(cls_scores, bbox_preds,
                                                 [gt_instances], img_metas)
        onegt_cls_loss = sum(one_gt_losses['loss_cls'])
        onegt_box_loss = sum(one_gt_losses['loss_bbox'])
        self.assertGreater(onegt_cls_loss.item(), 0,
                           'cls loss should be non-zero')
        self.assertGreater(onegt_box_loss.item(), 0,
                           'box loss should be non-zero')
