# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
from mmengine.structures import InstanceData

from mmdet.models.dense_heads import CentripetalHead


class TestCentripetalHead(TestCase):

    def test_centripetal_head_loss(self):
        """Tests corner head loss when truth is empty and non-empty."""
        s = 256
        img_metas = [{
            'img_shape': (s, s, 3),
            'scale_factor': 1,
            'batch_input_shape': (s, s, 3)
        }]

        centripetal_head = CentripetalHead(
            num_classes=4, in_channels=1, corner_emb_channels=0)

        # Corner head expects a multiple levels of features per image
        feat = [
            torch.rand(1, 1, s // 4, s // 4)
            for _ in range(centripetal_head.num_feat_levels)
        ]
        forward_outputs = centripetal_head.forward(feat)

        # Test that empty ground truth encourages the network
        # to predict background
        gt_instances = InstanceData()
        gt_instances.bboxes = torch.empty((0, 4))
        gt_instances.labels = torch.LongTensor([])
        gt_bboxes_ignore = None

        empty_gt_losses = centripetal_head.loss_by_feat(
            *forward_outputs, [gt_instances], img_metas, gt_bboxes_ignore)
        empty_det_loss = sum(empty_gt_losses['det_loss'])
        empty_guiding_loss = sum(empty_gt_losses['guiding_loss'])
        empty_centripetal_loss = sum(empty_gt_losses['centripetal_loss'])
        empty_off_loss = sum(empty_gt_losses['off_loss'])
        self.assertTrue(empty_det_loss.item() > 0,
                        'det loss should be non-zero')
        self.assertTrue(
            empty_guiding_loss.item() == 0,
            'there should be no guiding loss when there are no true boxes')
        self.assertTrue(
            empty_centripetal_loss.item() == 0,
            'there should be no centripetal loss when there are no true boxes')
        self.assertTrue(
            empty_off_loss.item() == 0,
            'there should be no box loss when there are no true boxes')

        gt_instances = InstanceData()
        gt_instances.bboxes = torch.Tensor(
            [[23.6667, 23.8757, 238.6326, 151.8874],
             [123.6667, 123.8757, 138.6326, 251.8874]])
        gt_instances.labels = torch.LongTensor([2, 3])

        two_gt_losses = centripetal_head.loss_by_feat(*forward_outputs,
                                                      [gt_instances],
                                                      img_metas,
                                                      gt_bboxes_ignore)
        twogt_det_loss = sum(two_gt_losses['det_loss'])
        twogt_guiding_loss = sum(two_gt_losses['guiding_loss'])
        twogt_centripetal_loss = sum(two_gt_losses['centripetal_loss'])
        twogt_off_loss = sum(two_gt_losses['off_loss'])
        assert twogt_det_loss.item() > 0, 'det loss should be non-zero'
        assert twogt_guiding_loss.item() > 0, 'push loss should be non-zero'
        assert twogt_centripetal_loss.item(
        ) > 0, 'pull loss should be non-zero'
        assert twogt_off_loss.item() > 0, 'off loss should be non-zero'
