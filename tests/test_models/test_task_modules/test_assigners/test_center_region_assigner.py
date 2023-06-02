# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
from mmengine.structures import InstanceData

from mmdet.models.task_modules.assigners import CenterRegionAssigner


class TestCenterRegionAssigner(TestCase):

    def test_center_region_assigner(self):
        center_region_assigner = CenterRegionAssigner(
            pos_scale=0.2, neg_scale=0.2, min_pos_iof=0.01)
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

        assign_result = center_region_assigner.assign(pred_instances,
                                                      gt_instances)
        self.assertEqual(len(assign_result.gt_inds), 4)
        self.assertEqual(len(assign_result.labels), 4)

        expected_gt_inds = torch.LongTensor([1, 0, 0, 0])
        self.assertTrue(torch.all(assign_result.gt_inds == expected_gt_inds))
        expected_shadowed_labels = torch.LongTensor([[2, 3]])
        shadowed_labels = assign_result.get_extra_property('shadowed_labels')
        self.assertTrue(torch.all(shadowed_labels == expected_shadowed_labels))

    def test_center_region_assigner_with_ignore(self):
        center_region_assigner = CenterRegionAssigner(
            pos_scale=0.2, neg_scale=0.2, min_pos_iof=0.01)
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
        assign_result = center_region_assigner.assign(
            pred_instances,
            gt_instances,
            gt_instances_ignore=gt_instances_ignore)

        expected_gt_inds = torch.LongTensor([1, 0, 0, -1])
        self.assertTrue(torch.all(assign_result.gt_inds == expected_gt_inds))
        self.assertTrue(torch.all(assign_result.gt_inds == expected_gt_inds))

    def test_center_region_assigner_with_empty_gt(self):
        """Test corner case where an image might have no true detections."""
        center_region_assigner = CenterRegionAssigner(
            pos_scale=0.2, neg_scale=0.2, min_pos_iof=0.01)
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
        assign_result = center_region_assigner.assign(pred_instances,
                                                      gt_instances)

        expected_gt_inds = torch.LongTensor([0, 0, 0, 0])
        self.assertTrue(torch.all(assign_result.gt_inds == expected_gt_inds))

    def test_center_region_assigner_with_empty_boxes(self):
        """Test corner case where a network might predict no boxes."""
        center_region_assigner = CenterRegionAssigner(
            pos_scale=0.2, neg_scale=0.2, min_pos_iof=0.01)
        priors = torch.empty((0, 4))
        gt_bboxes = torch.FloatTensor([
            [0, 0, 10, 9],
            [0, 10, 10, 19],
        ])
        gt_labels = torch.LongTensor([2, 3])

        pred_instances = InstanceData(priors=priors)
        gt_instances = InstanceData(bboxes=gt_bboxes, labels=gt_labels)
        assign_result = center_region_assigner.assign(pred_instances,
                                                      gt_instances)
        self.assertEqual(len(assign_result.gt_inds), 0)
        self.assertTrue(tuple(assign_result.labels.shape) == (0, ))

    def test_center_region_assigner_with_empty_boxes_and_ignore(self):
        """Test corner case where a network might predict no boxes and
        ignore_iof_thr is on."""
        center_region_assigner = CenterRegionAssigner(
            pos_scale=0.2, neg_scale=0.2, min_pos_iof=0.01)
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

        assign_result = center_region_assigner.assign(
            pred_instances,
            gt_instances,
            gt_instances_ignore=gt_instances_ignore)
        self.assertEqual(len(assign_result.gt_inds), 0)
        self.assertTrue(tuple(assign_result.labels.shape) == (0, ))

    def test_center_region_assigner_with_empty_boxes_and_gt(self):
        """Test corner case where a network might predict no boxes and no
        gt."""
        center_region_assigner = CenterRegionAssigner(
            pos_scale=0.2, neg_scale=0.2, min_pos_iof=0.01)
        priors = torch.empty((0, 4))
        gt_bboxes = torch.empty((0, 4))
        gt_labels = torch.empty(0)

        pred_instances = InstanceData(priors=priors)
        gt_instances = InstanceData(bboxes=gt_bboxes, labels=gt_labels)
        assign_result = center_region_assigner.assign(pred_instances,
                                                      gt_instances)
        self.assertEqual(len(assign_result.gt_inds), 0)
