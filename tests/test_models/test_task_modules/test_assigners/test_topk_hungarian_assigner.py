# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
from mmengine import ConfigDict

from mmdet.models.task_modules.assigners import TopkHungarianAssigner


class TestTopkHungarianAssigner(TestCase):

    def test_init(self):
        with self.assertRaises(AssertionError):
            TopkHungarianAssigner(topk=0)

    def test_topk_hungarian_match_assigner(self):
        num_classes = 80
        topk = 4

        pred_class_scores = torch.rand((10, num_classes))
        pred_bboxes = torch.randn((10, 4))

        assigner = TopkHungarianAssigner(topk=topk)

        # test no gt bboxes
        gt_bboxes = torch.empty((0, 4))
        gt_labels = torch.LongTensor([])
        img_meta = dict(img_shape=(10, 8))

        assign_result = assigner.assign(pred_class_scores, pred_bboxes,
                                        gt_bboxes, gt_labels, img_meta)

        self.assertTrue(torch.all(assign_result.gt_inds == 0))
        self.assertTrue(torch.all(assign_result.labels == -1))

        # test with gt bboxes
        gt_bboxes = torch.FloatTensor([[0, 0, 5, 7], [3, 5, 7, 8]])
        gt_labels = torch.LongTensor([1, 20])
        assign_result = assigner.assign(pred_class_scores, pred_bboxes,
                                        gt_bboxes, gt_labels, img_meta)

        self.assertTrue(torch.all(assign_result.gt_inds > -1))
        self.assertEqual((assign_result.gt_inds > 0).sum(),
                         gt_bboxes.size(0) * topk)
        self.assertEqual((assign_result.labels > -1).sum(),
                         gt_bboxes.size(0) * topk)

    def test_bbox_match_cost(self):
        num_classes = 80
        topk = 4

        pred_class_scores = torch.rand((10, num_classes))
        pred_bboxes = torch.randn((10, 4))

        gt_bboxes = torch.FloatTensor([[0, 0, 5, 7], [3, 5, 7, 8]])
        gt_labels = torch.LongTensor([1, 20])
        img_meta = dict(img_shape=(10, 8))

        # test IoUCost
        assigner = TopkHungarianAssigner(
            topk=topk,
            iou_cost=ConfigDict(dict(type='IoUCost', iou_mode='iou')))
        assign_result = assigner.assign(pred_class_scores, pred_bboxes,
                                        gt_bboxes, gt_labels, img_meta)

        self.assertTrue(torch.all(assign_result.gt_inds > -1))
        self.assertEqual((assign_result.gt_inds > 0).sum(),
                         gt_bboxes.size(0) * topk)
        self.assertEqual((assign_result.labels > -1).sum(),
                         gt_bboxes.size(0) * topk)

        # test BBoxL1Cost
        assigner = TopkHungarianAssigner(
            topk=4, reg_cost=ConfigDict(dict(type='BBoxL1Cost')))
        assign_result = assigner.assign(pred_class_scores, pred_bboxes,
                                        gt_bboxes, gt_labels, img_meta)
        self.assertTrue(torch.all(assign_result.gt_inds > -1))
        self.assertEqual((assign_result.gt_inds > 0).sum(),
                         gt_bboxes.size(0) * topk)
        self.assertEqual((assign_result.labels > -1).sum(),
                         gt_bboxes.size(0) * topk)

    def test_cls_match_cost(self):
        num_classes = 80
        topk = 4

        pred_class_scores = torch.rand((10, num_classes))
        pred_bboxes = torch.randn((10, 4))

        gt_bboxes = torch.FloatTensor([[0, 0, 5, 7], [3, 5, 7, 8]])
        gt_labels = torch.LongTensor([1, 20])
        img_meta = dict(img_shape=(10, 8))

        # test FocalLossCost
        assigner = TopkHungarianAssigner(
            topk=topk, cls_cost=dict(type='FocalLossCost'))
        assign_result = assigner.assign(pred_class_scores, pred_bboxes,
                                        gt_bboxes, gt_labels, img_meta)
        self.assertTrue(torch.all(assign_result.gt_inds > -1))
        self.assertEqual((assign_result.gt_inds > 0).sum(),
                         gt_bboxes.size(0) * topk)
        self.assertEqual((assign_result.labels > -1).sum(),
                         gt_bboxes.size(0) * topk)

        # test ClassificationCost
        assigner = TopkHungarianAssigner(
            topk=4, cls_cost=dict(type='ClassificationCost'))
        assign_result = assigner.assign(pred_class_scores, pred_bboxes,
                                        gt_bboxes, gt_labels, img_meta)
        self.assertTrue(torch.all(assign_result.gt_inds > -1))
        self.assertEqual((assign_result.gt_inds > 0).sum(),
                         gt_bboxes.size(0) * topk)
        self.assertEqual((assign_result.labels > -1).sum(),
                         gt_bboxes.size(0) * topk)
