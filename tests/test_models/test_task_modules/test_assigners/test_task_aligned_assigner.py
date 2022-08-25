# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
from mmengine.structures import InstanceData

from mmdet.models.task_modules.assigners import TaskAlignedAssigner


class TestTaskAlignedAssigner(TestCase):

    def test_task_aligned_assigner(self):

        with self.assertRaises(AssertionError):
            TaskAlignedAssigner(topk=0)

        assigner = TaskAlignedAssigner(topk=13)
        pred_score = torch.FloatTensor([[0.1, 0.2], [0.2, 0.3], [0.3, 0.4],
                                        [0.4, 0.5]])
        pred_bbox = torch.FloatTensor([
            [1, 1, 12, 8],
            [4, 4, 20, 20],
            [1, 5, 15, 15],
            [30, 5, 32, 42],
        ])
        anchor = torch.FloatTensor([
            [0, 0, 10, 10],
            [10, 10, 20, 20],
            [5, 5, 15, 15],
            [32, 32, 38, 42],
        ])
        gt_bboxes = torch.FloatTensor([
            [0, 0, 10, 9],
            [0, 10, 10, 19],
        ])
        gt_labels = torch.LongTensor([0, 1])
        pred_instances = InstanceData()
        pred_instances.priors = anchor
        pred_instances.bboxes = pred_bbox
        pred_instances.scores = pred_score
        gt_instances = InstanceData()
        gt_instances.bboxes = gt_bboxes
        gt_instances.labels = gt_labels

        assign_result = assigner.assign(pred_instances, gt_instances)
        self.assertEqual(len(assign_result.gt_inds), 4)
        self.assertEqual(len(assign_result.labels), 4)

        # test empty gt
        gt_bboxes = torch.empty(0, 4)
        gt_labels = torch.empty(0, 2).long()
        gt_instances = InstanceData()
        gt_instances.bboxes = gt_bboxes
        gt_instances.labels = gt_labels
        assign_result = assigner.assign(pred_instances, gt_instances)
        expected_gt_inds = torch.LongTensor([0, 0, 0, 0])
        self.assertTrue(torch.all(assign_result.gt_inds == expected_gt_inds))
