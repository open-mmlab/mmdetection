# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
from mmengine.structures import InstanceData

from mmdet.models.dense_heads import NASFCOSHead


class TestNASFCOSHead(TestCase):

    def test_nasfcos_head_loss(self):
        """Tests nasfcos head loss when truth is empty and non-empty."""
        s = 256
        img_metas = [{
            'img_shape': (s, s, 3),
            'pad_shape': (s, s, 3),
            'scale_factor': 1,
        }]
        nasfcos_head = NASFCOSHead(
            num_classes=4,
            in_channels=2,  # the same as `deform_groups` in dconv3x3_config
            feat_channels=2,
            norm_cfg=None)

        # Nasfcos head expects a multiple levels of features per image
        feats = (
            torch.rand(1, 2, s // stride[1], s // stride[0]).float()
            for stride in nasfcos_head.prior_generator.strides)
        cls_scores, bbox_preds, centernesses = nasfcos_head.forward(feats)

        # Test that empty ground truth encourages the network to
        # predict background
        gt_instances = InstanceData()
        gt_instances.bboxes = torch.empty((0, 4))
        gt_instances.labels = torch.LongTensor([])

        empty_gt_losses = nasfcos_head.loss_by_feat(cls_scores, bbox_preds,
                                                    centernesses,
                                                    [gt_instances], img_metas)
        # When there is no truth, the cls loss should be nonzero but
        # box loss and centerness loss should be zero
        empty_cls_loss = empty_gt_losses['loss_cls'].item()
        empty_box_loss = empty_gt_losses['loss_bbox'].item()
        empty_ctr_loss = empty_gt_losses['loss_centerness'].item()
        self.assertGreater(empty_cls_loss, 0, 'cls loss should be non-zero')
        self.assertEqual(
            empty_box_loss, 0,
            'there should be no box loss when there are no true boxes')
        self.assertEqual(
            empty_ctr_loss, 0,
            'there should be no centerness loss when there are no true boxes')

        # When truth is non-empty then all cls, box loss and centerness loss
        # should be nonzero for random inputs
        gt_instances = InstanceData()
        gt_instances.bboxes = torch.Tensor(
            [[23.6667, 23.8757, 238.6326, 151.8874]])
        gt_instances.labels = torch.LongTensor([2])

        one_gt_losses = nasfcos_head.loss_by_feat(cls_scores, bbox_preds,
                                                  centernesses, [gt_instances],
                                                  img_metas)
        onegt_cls_loss = one_gt_losses['loss_cls'].item()
        onegt_box_loss = one_gt_losses['loss_bbox'].item()
        onegt_ctr_loss = one_gt_losses['loss_centerness'].item()
        self.assertGreater(onegt_cls_loss, 0, 'cls loss should be non-zero')
        self.assertGreater(onegt_box_loss, 0, 'box loss should be non-zero')
        self.assertGreater(onegt_ctr_loss, 0,
                           'centerness loss should be non-zero')
