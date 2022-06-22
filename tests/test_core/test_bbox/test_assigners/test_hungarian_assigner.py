# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
from mmengine.data import InstanceData

from mmdet.core.bbox.assigners import HungarianAssigner


class TestHungrianMatchAssigner(TestCase):

    def test_hungarian_match_assigner(self):
        assigner = HungarianAssigner()
        self.assertEqual(assigner.iou_cost.iou_mode, 'giou')

        # test no gt bboxes
        gt_instances = InstanceData()
        gt_instances.bboxes = torch.empty((0, 4)).float()
        gt_instances.labels = torch.empty((0, )).long()
        pred_instances = InstanceData()
        pred_instances.scores = torch.rand((10, 81))
        pred_instances.bboxes = torch.rand((10, 4))
        img_meta = dict(img_shape=(10, 8))

        assign_result = assigner.assign(
            pred_instances, gt_instances, img_meta=img_meta)
        self.assertTrue(torch.all(assign_result.gt_inds == 0))
        self.assertTrue(torch.all(assign_result.labels == -1))

        # test with gt bboxes
        gt_instances.bboxes = torch.FloatTensor([[0, 0, 5, 7], [3, 5, 7, 8]])
        gt_instances.labels = torch.LongTensor([1, 20])
        assign_result = assigner.assign(
            pred_instances, gt_instances, img_meta=img_meta)

        self.assertTrue(torch.all(assign_result.gt_inds > -1))
        self.assertEqual((assign_result.gt_inds > 0).sum(),
                         gt_instances.bboxes.size(0))
        self.assertEqual((assign_result.labels > -1).sum(),
                         gt_instances.bboxes.size(0))

        # test iou mode
        assigner = HungarianAssigner(
            iou_cost=dict(type='IoUCost', iou_mode='iou', weight=1.0))
        self.assertEqual(assigner.iou_cost.iou_mode, 'iou')
        assign_result = assigner.assign(
            pred_instances, gt_instances, img_meta=img_meta)
        self.assertTrue(torch.all(assign_result.gt_inds > -1))
        self.assertEqual((assign_result.gt_inds > 0).sum(),
                         gt_instances.bboxes.size(0))
        self.assertEqual((assign_result.labels > -1).sum(),
                         gt_instances.bboxes.size(0))

        # test focal loss mode
        assigner = HungarianAssigner(
            iou_cost=dict(type='IoUCost', iou_mode='giou', weight=1.0),
            cls_cost=dict(type='FocalLossCost', weight=1.))
        self.assertEqual(assigner.iou_cost.iou_mode, 'giou')
        assign_result = assigner.assign(
            pred_instances, gt_instances, img_meta=img_meta)
        self.assertTrue(torch.all(assign_result.gt_inds > -1))
        self.assertEqual((assign_result.gt_inds > 0).sum(),
                         gt_instances.bboxes.size(0))
        self.assertEqual((assign_result.labels > -1).sum(),
                         gt_instances.bboxes.size(0))
