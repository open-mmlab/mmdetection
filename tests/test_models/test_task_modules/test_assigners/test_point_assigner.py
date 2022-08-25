# Copyright (c) OpenMMLab. All rights reserved.
import unittest

import torch
from mmengine.structures import InstanceData
from mmengine.testing import assert_allclose

from mmdet.models.task_modules.assigners import PointAssigner


class TestPointAssigner(unittest.TestCase):

    def test_point_assigner(self):
        assigner = PointAssigner()
        pred_instances = InstanceData()
        pred_instances.priors = torch.FloatTensor([
            # [x, y, stride]
            [0, 0, 1],
            [10, 10, 1],
            [5, 5, 1],
            [32, 32, 1],
        ])
        gt_instances = InstanceData()
        gt_instances.bboxes = torch.FloatTensor([
            [0, 0, 10, 9],
            [0, 10, 10, 19],
        ])
        gt_instances.labels = torch.LongTensor([0, 1])
        assign_result = assigner.assign(pred_instances, gt_instances)
        expected_gt_inds = torch.LongTensor([1, 2, 1, 0])
        assert_allclose(assign_result.gt_inds, expected_gt_inds)

    def test_point_assigner_with_empty_gt(self):
        """Test corner case where an image might have no true detections."""
        assigner = PointAssigner()
        pred_instances = InstanceData()
        pred_instances.priors = torch.FloatTensor([
            # [x, y, stride]
            [0, 0, 1],
            [10, 10, 1],
            [5, 5, 1],
            [32, 32, 1],
        ])
        gt_instances = InstanceData()
        gt_instances.bboxes = torch.FloatTensor([])
        gt_instances.labels = torch.LongTensor([])
        assign_result = assigner.assign(pred_instances, gt_instances)

        expected_gt_inds = torch.LongTensor([0, 0, 0, 0])
        assert_allclose(assign_result.gt_inds, expected_gt_inds)

    def test_point_assigner_with_empty_boxes_and_gt(self):
        """Test corner case where an image might predict no points and no
        gt."""
        assigner = PointAssigner()
        pred_instances = InstanceData()
        pred_instances.priors = torch.FloatTensor([])
        gt_instances = InstanceData()
        gt_instances.bboxes = torch.FloatTensor([])
        gt_instances.labels = torch.LongTensor([])
        assign_result = assigner.assign(pred_instances, gt_instances)
        self.assertEqual(len(assign_result.gt_inds), 0)
