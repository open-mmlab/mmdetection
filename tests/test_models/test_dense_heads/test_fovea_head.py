# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
from mmengine.structures import InstanceData

from mmdet.models.dense_heads import FoveaHead


class TestFOVEAHead(TestCase):

    def test_fovea_head_loss(self):
        """Tests anchor head loss when truth is empty and non-empty."""
        s = 256
        img_metas = [{
            'img_shape': (s, s, 3),
            'pad_shape': (s, s, 3),
            'scale_factor': 1,
        }]

        fovea_head = FoveaHead(num_classes=4, in_channels=1)

        # Anchor head expects a multiple levels of features per image
        feats = (
            torch.rand(1, 1, s // (2**(i + 2)), s // (2**(i + 2)))
            for i in range(len(fovea_head.prior_generator.strides)))
        cls_scores, bbox_preds = fovea_head.forward(feats)

        # Test that empty ground truth encourages the network to
        # predict background
        gt_instances = InstanceData()
        gt_instances.bboxes = torch.empty((0, 4))
        gt_instances.labels = torch.LongTensor([])

        empty_gt_losses = fovea_head.loss_by_feat(cls_scores, bbox_preds,
                                                  [gt_instances], img_metas)
        # When there is no truth, the cls loss should be nonzero but
        # there should be no box loss.
        empty_cls_loss = empty_gt_losses['loss_cls']
        empty_box_loss = empty_gt_losses['loss_bbox']
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

        one_gt_losses = fovea_head.loss_by_feat(cls_scores, bbox_preds,
                                                [gt_instances], img_metas)
        onegt_cls_loss = one_gt_losses['loss_cls']
        onegt_box_loss = one_gt_losses['loss_bbox']
        self.assertGreater(onegt_cls_loss.item(), 0,
                           'cls loss should be non-zero')
        self.assertGreater(onegt_box_loss.item(), 0,
                           'box loss should be non-zero')
