# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
from mmengine.structures import InstanceData

from mmdet import *  # noqa
from mmdet.models.dense_heads import FreeAnchorRetinaHead


class TestFreeAnchorRetinaHead(TestCase):

    def test_free_anchor_head_loss(self):
        """Tests rpn head loss when truth is empty and non-empty."""
        s = 256
        img_metas = [{
            'img_shape': (s, s, 3),
            'pad_shape': (s, s, 3),
            'scale_factor': 1,
        }]

        anchor_head = FreeAnchorRetinaHead(num_classes=1, in_channels=1)

        # Anchor head expects a multiple levels of features per image
        feats = (
            torch.rand(1, 1, s // (2**(i + 2)), s // (2**(i + 2)))
            for i in range(len(anchor_head.prior_generator.strides)))
        cls_scores, bbox_preds = anchor_head.forward(feats)

        # Test that empty ground truth encourages the network to
        # predict background
        gt_instances = InstanceData()
        gt_instances.bboxes = torch.empty((0, 4))
        gt_instances.labels = torch.LongTensor([])

        empty_gt_losses = anchor_head.loss_by_feat(cls_scores, bbox_preds,
                                                   [gt_instances], img_metas)
        # When there is no truth, the cls loss should be nonzero but
        # there should be no box loss.
        positive_bag_loss = empty_gt_losses['positive_bag_loss']
        negative_bag_loss = empty_gt_losses['negative_bag_loss']
        self.assertGreater(negative_bag_loss.item(), 0,
                           'negative_bag loss should be non-zero')
        self.assertEqual(
            positive_bag_loss.item(), 0,
            'there should be no positive_bag loss when there are no true boxes'
        )

        # When truth is non-empty then both cls and box loss
        # should be nonzero for random inputs
        gt_instances = InstanceData()
        gt_instances.bboxes = torch.Tensor(
            [[23.6667, 23.8757, 238.6326, 151.8874]])
        gt_instances.labels = torch.LongTensor([0])

        one_gt_losses = anchor_head.loss_by_feat(cls_scores, bbox_preds,
                                                 [gt_instances], img_metas)
        onegt_cls_loss = one_gt_losses['positive_bag_loss']
        onegt_box_loss = one_gt_losses['negative_bag_loss']
        self.assertGreater(onegt_cls_loss.item(), 0,
                           'positive bag loss should be non-zero')
        self.assertGreater(onegt_box_loss.item(), 0,
                           'negative bag loss should be non-zero')
