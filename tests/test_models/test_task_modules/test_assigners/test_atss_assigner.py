# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
from mmengine.structures import InstanceData

from mmdet.models.task_modules.assigners import ATSSAssigner


class TestATSSAssigner(TestCase):

    def test_atss_assigner(self):
        atss_assigner = ATSSAssigner(topk=9)
        priors = torch.FloatTensor([
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

        pred_instances = InstanceData(priors=priors)
        gt_instances = InstanceData(bboxes=gt_bboxes, labels=gt_labels)
        num_level_bboxes = [4]

        assign_result = atss_assigner.assign(pred_instances, num_level_bboxes,
                                             gt_instances)
        self.assertEqual(len(assign_result.gt_inds), 4)
        self.assertEqual(len(assign_result.labels), 4)

        expected_gt_inds = torch.LongTensor([1, 0, 0, 0])
        self.assertTrue(torch.all(assign_result.gt_inds == expected_gt_inds))

    def test_atss_assigner_with_ignore(self):
        atss_assigner = ATSSAssigner(topk=9)
        priors = torch.FloatTensor([
            [0, 0, 10, 10],
            [10, 10, 20, 20],
            [5, 5, 15, 15],
            [30, 32, 40, 42],
        ])
        gt_bboxes = torch.FloatTensor([
            [0, 0, 10, 9],
            [0, 10, 10, 19],
        ])
        gt_labels = torch.LongTensor([2, 3])
        gt_bboxes_ignore = torch.Tensor([
            [30, 30, 40, 40],
        ])

        pred_instances = InstanceData(priors=priors)
        gt_instances = InstanceData(bboxes=gt_bboxes, labels=gt_labels)
        gt_instances_ignore = InstanceData(bboxes=gt_bboxes_ignore)
        num_level_bboxes = [4]
        assign_result = atss_assigner.assign(
            pred_instances,
            num_level_bboxes,
            gt_instances,
            gt_instances_ignore=gt_instances_ignore)

        expected_gt_inds = torch.LongTensor([1, 0, 0, 0])
        self.assertTrue(torch.all(assign_result.gt_inds == expected_gt_inds))

    def test_atss_assigner_with_empty_gt(self):
        """Test corner case where an image might have no true detections."""
        atss_assigner = ATSSAssigner(topk=9)
        priors = torch.FloatTensor([
            [0, 0, 10, 10],
            [10, 10, 20, 20],
            [5, 5, 15, 15],
            [32, 32, 38, 42],
        ])
        gt_bboxes = torch.empty(0, 4)
        gt_labels = torch.empty(0)

        pred_instances = InstanceData(priors=priors)
        gt_instances = InstanceData(bboxes=gt_bboxes, labels=gt_labels)
        num_level_bboxes = [4]
        assign_result = atss_assigner.assign(pred_instances, num_level_bboxes,
                                             gt_instances)

        expected_gt_inds = torch.LongTensor([0, 0, 0, 0])
        self.assertTrue(torch.all(assign_result.gt_inds == expected_gt_inds))

    def test_atss_assigner_with_empty_boxes(self):
        """Test corner case where a network might predict no boxes."""
        atss_assigner = ATSSAssigner(topk=9)
        priors = torch.empty((0, 4))
        gt_bboxes = torch.FloatTensor([
            [0, 0, 10, 9],
            [0, 10, 10, 19],
        ])
        gt_labels = torch.LongTensor([2, 3])

        pred_instances = InstanceData(priors=priors)
        gt_instances = InstanceData(bboxes=gt_bboxes, labels=gt_labels)
        num_level_bboxes = [0]
        assign_result = atss_assigner.assign(pred_instances, num_level_bboxes,
                                             gt_instances)
        self.assertEqual(len(assign_result.gt_inds), 0)
        self.assertTrue(tuple(assign_result.labels.shape) == (0, ))

    def test_atss_assigner_with_empty_boxes_and_ignore(self):
        """Test corner case where a network might predict no boxes and
        ignore_iof_thr is on."""
        atss_assigner = ATSSAssigner(topk=9)
        priors = torch.empty((0, 4))
        gt_bboxes = torch.FloatTensor([
            [0, 0, 10, 9],
            [0, 10, 10, 19],
        ])
        gt_bboxes_ignore = torch.Tensor([
            [30, 30, 40, 40],
        ])
        gt_labels = torch.LongTensor([2, 3])

        pred_instances = InstanceData(priors=priors)
        gt_instances = InstanceData(bboxes=gt_bboxes, labels=gt_labels)
        gt_instances_ignore = InstanceData(bboxes=gt_bboxes_ignore)
        num_level_bboxes = [0]

        assign_result = atss_assigner.assign(
            pred_instances,
            num_level_bboxes,
            gt_instances,
            gt_instances_ignore=gt_instances_ignore)
        self.assertEqual(len(assign_result.gt_inds), 0)
        self.assertTrue(tuple(assign_result.labels.shape) == (0, ))

    def test_atss_assigner_with_empty_boxes_and_gt(self):
        """Test corner case where a network might predict no boxes and no
        gt."""
        atss_assigner = ATSSAssigner(topk=9)
        priors = torch.empty((0, 4))
        gt_bboxes = torch.empty((0, 4))
        gt_labels = torch.empty(0)
        num_level_bboxes = [0]

        pred_instances = InstanceData(priors=priors)
        gt_instances = InstanceData(bboxes=gt_bboxes, labels=gt_labels)
        assign_result = atss_assigner.assign(pred_instances, num_level_bboxes,
                                             gt_instances)
        self.assertEqual(len(assign_result.gt_inds), 0)
