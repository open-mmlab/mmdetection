# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
from mmengine.structures import InstanceData

from mmdet.models.task_modules.assigners import ApproxMaxIoUAssigner


class TestApproxIoUAssigner(TestCase):

    def test_approx_iou_assigner(self):
        assigner = ApproxMaxIoUAssigner(
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
        )
        bboxes = torch.FloatTensor([
            [0, 0, 10, 10],
            [10, 10, 20, 20],
            [5, 5, 15, 15],
            [32, 32, 38, 42],
        ])
        gt_bboxes = torch.FloatTensor([
            [0, 0, 10, 9],
            [0, 10, 10, 19],
        ])
        gt_labels = torch.LongTensor([2, 3])

        pred_instances = InstanceData()
        pred_instances.priors = bboxes
        pred_instances.approxs = bboxes[:, None, :]
        gt_instances = InstanceData()
        gt_instances.bboxes = gt_bboxes
        gt_instances.labels = gt_labels
        assign_result = assigner.assign(pred_instances, gt_instances)

        expected_gt_inds = torch.LongTensor([1, 0, 2, 0])
        self.assertTrue(torch.all(assign_result.gt_inds == expected_gt_inds))

    def test_approx_iou_assigner_with_empty_gt(self):
        """Test corner case where an image might have no true detections."""
        assigner = ApproxMaxIoUAssigner(
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
        )
        bboxes = torch.FloatTensor([
            [0, 0, 10, 10],
            [10, 10, 20, 20],
            [5, 5, 15, 15],
            [32, 32, 38, 42],
        ])
        gt_bboxes = torch.FloatTensor([])
        gt_labels = torch.LongTensor([])

        pred_instances = InstanceData()
        pred_instances.priors = bboxes
        pred_instances.approxs = bboxes[:, None, :]
        gt_instances = InstanceData()
        gt_instances.bboxes = gt_bboxes
        gt_instances.labels = gt_labels
        assign_result = assigner.assign(pred_instances, gt_instances)

        expected_gt_inds = torch.LongTensor([0, 0, 0, 0])
        self.assertTrue(torch.all(assign_result.gt_inds == expected_gt_inds))

    def test_approx_iou_assigner_with_empty_boxes(self):
        """Test corner case where an network might predict no boxes."""
        assigner = ApproxMaxIoUAssigner(
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
        )
        bboxes = torch.empty((0, 4))
        gt_bboxes = torch.FloatTensor([
            [0, 0, 10, 9],
            [0, 10, 10, 19],
        ])
        gt_labels = torch.LongTensor([2, 3])

        pred_instances = InstanceData()
        pred_instances.priors = bboxes
        pred_instances.approxs = bboxes[:, None, :]
        gt_instances = InstanceData()
        gt_instances.bboxes = gt_bboxes
        gt_instances.labels = gt_labels
        assign_result = assigner.assign(pred_instances, gt_instances)

        self.assertEqual(len(assign_result.gt_inds), 0)

    def test_approx_iou_assigner_with_empty_boxes_and_gt(self):
        """Test corner case where an network might predict no boxes and no
        gt."""
        assigner = ApproxMaxIoUAssigner(
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
        )
        bboxes = torch.empty((0, 4))
        gt_bboxes = torch.empty((0, 4))
        gt_labels = torch.LongTensor([])

        pred_instances = InstanceData()
        pred_instances.priors = bboxes
        pred_instances.approxs = bboxes[:, None, :]
        gt_instances = InstanceData()
        gt_instances.bboxes = gt_bboxes
        gt_instances.labels = gt_labels
        assign_result = assigner.assign(pred_instances, gt_instances)

        self.assertEqual(len(assign_result.gt_inds), 0)
