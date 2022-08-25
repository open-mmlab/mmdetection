# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
from mmengine.structures import InstanceData
from mmengine.testing import assert_allclose

from mmdet.models.task_modules.assigners import UniformAssigner


class TestUniformAssigner(TestCase):

    def test_uniform_assigner(self):
        assigner = UniformAssigner(0.15, 0.7, 1)
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
        gt_labels = torch.LongTensor([2, 3])
        pred_instances = InstanceData()
        pred_instances.priors = anchor
        pred_instances.decoder_priors = pred_bbox
        gt_instances = InstanceData()
        gt_instances.bboxes = gt_bboxes
        gt_instances.labels = gt_labels
        assign_result = assigner.assign(pred_instances, gt_instances)
        self.assertEqual(len(assign_result.gt_inds), 4)
        self.assertEqual(len(assign_result.labels), 4)

        expected_gt_inds = torch.LongTensor([-1, 0, 2, 0])
        assert_allclose(assign_result.gt_inds, expected_gt_inds)

    def test_uniform_assigner_with_empty_gt(self):
        """Test corner case where an image might have no true detections."""
        assigner = UniformAssigner(0.15, 0.7, 1)
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
        gt_bboxes = torch.empty(0, 4)
        gt_labels = torch.empty(0)
        pred_instances = InstanceData()
        pred_instances.priors = anchor
        pred_instances.decoder_priors = pred_bbox
        gt_instances = InstanceData()
        gt_instances.bboxes = gt_bboxes
        gt_instances.labels = gt_labels
        assign_result = assigner.assign(pred_instances, gt_instances)

        expected_gt_inds = torch.LongTensor([0, 0, 0, 0])
        assert_allclose(assign_result.gt_inds, expected_gt_inds)

    def test_uniform_assigner_with_empty_boxes(self):
        """Test corner case where a network might predict no boxes."""
        assigner = UniformAssigner(0.15, 0.7, 1)
        pred_bbox = torch.empty((0, 4))
        anchor = torch.empty((0, 4))
        gt_bboxes = torch.FloatTensor([
            [0, 0, 10, 9],
            [0, 10, 10, 19],
        ])
        gt_labels = torch.LongTensor([2, 3])
        pred_instances = InstanceData()
        pred_instances.priors = anchor
        pred_instances.decoder_priors = pred_bbox
        gt_instances = InstanceData()
        gt_instances.bboxes = gt_bboxes
        gt_instances.labels = gt_labels

        # Test with gt_labels
        assign_result = assigner.assign(pred_instances, gt_instances)
        self.assertEqual(len(assign_result.gt_inds), 0)
        self.assertEqual(tuple(assign_result.labels.shape), (0, ))
