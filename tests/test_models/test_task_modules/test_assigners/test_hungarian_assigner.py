# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
from mmengine import ConfigDict
from mmengine.structures import InstanceData

from mmdet.models.task_modules.assigners import HungarianAssigner


class TestHungarianAssigner(TestCase):

    def test_init(self):
        with self.assertRaises(AssertionError):
            HungarianAssigner([])

    def test_hungarian_match_assigner(self):
        assigner = HungarianAssigner([
            dict(type='ClassificationCost', weight=1.),
            dict(type='BBoxL1Cost', weight=5.0),
            dict(type='IoUCost', iou_mode='giou', weight=2.0)
        ])

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

    def test_bbox_match_cost(self):
        gt_instances = InstanceData()
        gt_instances.bboxes = torch.FloatTensor([[0, 0, 5, 7], [3, 5, 7, 8]])
        gt_instances.labels = torch.LongTensor([1, 20])
        pred_instances = InstanceData()
        pred_instances.scores = torch.rand((10, 81))
        pred_instances.bboxes = torch.rand((10, 4))
        img_meta = dict(img_shape=(10, 8))

        # test IoUCost
        assigner = HungarianAssigner(
            ConfigDict(dict(type='IoUCost', iou_mode='iou')))
        assign_result = assigner.assign(
            pred_instances, gt_instances, img_meta=img_meta)
        self.assertTrue(torch.all(assign_result.gt_inds > -1))
        self.assertEqual((assign_result.gt_inds > 0).sum(),
                         gt_instances.bboxes.size(0))
        self.assertEqual((assign_result.labels > -1).sum(),
                         gt_instances.bboxes.size(0))

        # test BBoxL1Cost
        assigner = HungarianAssigner(ConfigDict(dict(type='BBoxL1Cost')))
        assign_result = assigner.assign(
            pred_instances, gt_instances, img_meta=img_meta)
        self.assertTrue(torch.all(assign_result.gt_inds > -1))
        self.assertEqual((assign_result.gt_inds > 0).sum(),
                         gt_instances.bboxes.size(0))
        self.assertEqual((assign_result.labels > -1).sum(),
                         gt_instances.bboxes.size(0))

    def test_cls_match_cost(self):
        gt_instances = InstanceData()
        gt_instances.bboxes = torch.FloatTensor([[0, 0, 5, 7], [3, 5, 7, 8]])
        gt_instances.labels = torch.LongTensor([1, 20])
        pred_instances = InstanceData()
        pred_instances.scores = torch.rand((10, 81))
        pred_instances.bboxes = torch.rand((10, 4))
        img_meta = dict(img_shape=(10, 8))

        # test FocalLossCost
        assigner = HungarianAssigner(dict(type='FocalLossCost'))
        assign_result = assigner.assign(
            pred_instances, gt_instances, img_meta=img_meta)
        self.assertTrue(torch.all(assign_result.gt_inds > -1))
        self.assertEqual((assign_result.gt_inds > 0).sum(),
                         gt_instances.bboxes.size(0))
        self.assertEqual((assign_result.labels > -1).sum(),
                         gt_instances.bboxes.size(0))

        # test ClassificationCost
        assigner = HungarianAssigner(dict(type='ClassificationCost'))
        assign_result = assigner.assign(
            pred_instances, gt_instances, img_meta=img_meta)
        self.assertTrue(torch.all(assign_result.gt_inds > -1))
        self.assertEqual((assign_result.gt_inds > 0).sum(),
                         gt_instances.bboxes.size(0))
        self.assertEqual((assign_result.labels > -1).sum(),
                         gt_instances.bboxes.size(0))

    def test_mask_match_cost(self):
        gt_instances = InstanceData()
        gt_instances.masks = torch.randint(0, 2, (2, 10, 10)).long()
        gt_instances.labels = torch.LongTensor([1, 20])

        pred_instances = InstanceData()
        pred_instances.masks = torch.rand((4, 10, 10))
        pred_instances.scores = torch.rand((4, 25))
        img_meta = dict(img_shape=(10, 10))

        # test DiceCost
        assigner = HungarianAssigner(dict(type='DiceCost'))
        assign_result = assigner.assign(
            pred_instances, gt_instances, img_meta=img_meta)
        self.assertTrue(torch.all(assign_result.gt_inds > -1))
        self.assertEqual((assign_result.gt_inds > 0).sum(),
                         gt_instances.masks.size(0))
        self.assertEqual((assign_result.labels > -1).sum(),
                         gt_instances.masks.size(0))

        # test CrossEntropyLossCost
        assigner = HungarianAssigner(dict(type='CrossEntropyLossCost'))
        assign_result = assigner.assign(
            pred_instances, gt_instances, img_meta=img_meta)
        self.assertTrue(torch.all(assign_result.gt_inds > -1))
        self.assertEqual((assign_result.gt_inds > 0).sum(),
                         gt_instances.masks.size(0))
        self.assertEqual((assign_result.labels > -1).sum(),
                         gt_instances.masks.size(0))

        # test FocalLossCost
        assigner = HungarianAssigner(
            dict(type='FocalLossCost', binary_input=True))
        assign_result = assigner.assign(
            pred_instances, gt_instances, img_meta=img_meta)
        self.assertTrue(torch.all(assign_result.gt_inds > -1))
        self.assertEqual((assign_result.gt_inds > 0).sum(),
                         gt_instances.masks.size(0))
        self.assertEqual((assign_result.labels > -1).sum(),
                         gt_instances.masks.size(0))
