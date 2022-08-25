# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
from mmengine.structures import InstanceData

from mmdet.models.dense_heads import AutoAssignHead


class TestAutoAssignHead(TestCase):

    def test_autoassign_head_loss(self):
        """Tests autoassign head loss when truth is empty and non-empty."""
        s = 300
        img_metas = [{
            'img_shape': (s, s, 3),
            'pad_shape': (s, s, 3),
            'scale_factor': 1,
        }]

        autoassign_head = AutoAssignHead(
            num_classes=4,
            in_channels=1,
            stacked_convs=1,
            feat_channels=1,
            strides=[8, 16, 32, 64, 128],
            loss_bbox=dict(type='GIoULoss', loss_weight=5.0),
            norm_cfg=None)

        # Fcos head expects a multiple levels of features per image
        feats = (
            torch.rand(1, 1, s // stride[1], s // stride[0])
            for stride in autoassign_head.prior_generator.strides)
        cls_scores, bbox_preds, centernesses = autoassign_head.forward(feats)

        # Test that empty ground truth encourages the network to
        # predict background
        gt_instances = InstanceData()
        gt_instances.bboxes = torch.empty((0, 4))
        gt_instances.labels = torch.LongTensor([])

        empty_gt_losses = autoassign_head.loss_by_feat(cls_scores, bbox_preds,
                                                       centernesses,
                                                       [gt_instances],
                                                       img_metas)
        # When there is no truth, the neg loss should be nonzero but
        # pos loss and center loss should be zero
        empty_pos_loss = empty_gt_losses['loss_pos'].item()
        empty_neg_loss = empty_gt_losses['loss_neg'].item()
        empty_ctr_loss = empty_gt_losses['loss_center'].item()
        self.assertGreater(empty_neg_loss, 0, 'neg loss should be non-zero')
        self.assertEqual(
            empty_pos_loss, 0,
            'there should be no pos loss when there are no true boxes')
        self.assertEqual(
            empty_ctr_loss, 0,
            'there should be no centerness loss when there are no true boxes')

        # When truth is non-empty then all pos, neg loss and center loss
        # should be nonzero for random inputs
        gt_instances = InstanceData()
        gt_instances.bboxes = torch.Tensor(
            [[23.6667, 23.8757, 238.6326, 151.8874]])
        gt_instances.labels = torch.LongTensor([2])

        one_gt_losses = autoassign_head.loss_by_feat(cls_scores, bbox_preds,
                                                     centernesses,
                                                     [gt_instances], img_metas)
        onegt_pos_loss = one_gt_losses['loss_pos'].item()
        onegt_neg_loss = one_gt_losses['loss_neg'].item()
        onegt_ctr_loss = one_gt_losses['loss_center'].item()
        self.assertGreater(onegt_pos_loss, 0, 'pos loss should be non-zero')
        self.assertGreater(onegt_neg_loss, 0, 'neg loss should be non-zero')
        self.assertGreater(onegt_ctr_loss, 0, 'center loss should be non-zero')
