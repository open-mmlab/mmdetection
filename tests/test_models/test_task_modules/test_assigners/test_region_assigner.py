# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData

from mmdet.models.task_modules.assigners import RegionAssigner


class TestRegionAssigner(TestCase):

    def setUp(self):
        self.img_meta = ConfigDict(dict(img_shape=(256, 256)))
        self.featmap_sizes = [(64, 64)]
        self.anchor_scale = 10
        self.anchor_strides = [1]

    def test_region_assigner(self):
        region_assigner = RegionAssigner(center_ratio=0.5, ignore_ratio=0.8)
        priors = torch.FloatTensor([
            [0, 0, 10, 10],
            [10, 10, 20, 20],
            [5, 5, 15, 15],
            [32, 32, 38, 42],
        ])
        valid_flags = torch.BoolTensor([1, 1, 1, 1])
        gt_bboxes = torch.FloatTensor([
            [0, 0, 10, 9],
            [0, 10, 10, 19],
        ])
        gt_labels = torch.LongTensor([2, 3])

        pred_instances = InstanceData(priors=priors, valid_flags=valid_flags)
        gt_instances = InstanceData(bboxes=gt_bboxes, labels=gt_labels)
        num_level_anchors = [4]

        assign_result = region_assigner.assign(
            pred_instances, gt_instances, self.img_meta, self.featmap_sizes,
            num_level_anchors, self.anchor_scale, self.anchor_strides)
        self.assertEqual(len(assign_result.gt_inds), 4)
        self.assertEqual(len(assign_result.labels), 4)

        expected_gt_inds = torch.LongTensor([1, 0, 0, 0])
        self.assertTrue(torch.all(assign_result.gt_inds == expected_gt_inds))

    def test_region_assigner_with_ignore(self):
        region_assigner = RegionAssigner(center_ratio=0.5)
        priors = torch.FloatTensor([
            [0, 0, 10, 10],
            [10, 10, 20, 20],
            [5, 5, 15, 15],
            [30, 32, 40, 42],
        ])
        valid_flags = torch.BoolTensor([1, 1, 1, 1])
        gt_bboxes = torch.FloatTensor([
            [0, 0, 10, 9],
            [0, 10, 10, 19],
        ])
        gt_labels = torch.LongTensor([2, 3])
        gt_bboxes_ignore = torch.Tensor([
            [30, 30, 40, 40],
        ])

        pred_instances = InstanceData(priors=priors, valid_flags=valid_flags)
        gt_instances = InstanceData(bboxes=gt_bboxes, labels=gt_labels)
        gt_instances_ignore = InstanceData(bboxes=gt_bboxes_ignore)
        num_level_anchors = [4]
        with self.assertRaises(NotImplementedError):
            region_assigner.assign(
                pred_instances,
                gt_instances,
                self.img_meta,
                self.featmap_sizes,
                num_level_anchors,
                self.anchor_scale,
                self.anchor_strides,
                gt_instances_ignore=gt_instances_ignore)

    def test_region_assigner_with_empty_gt(self):
        """Test corner case where an image might have no true detections."""
        region_assigner = RegionAssigner(center_ratio=0.5)
        priors = torch.FloatTensor([
            [0, 0, 10, 10],
            [10, 10, 20, 20],
            [5, 5, 15, 15],
            [32, 32, 38, 42],
        ])
        valid_flags = torch.BoolTensor([1, 1, 1, 1])
        gt_bboxes = torch.empty(0, 4)
        gt_labels = torch.empty(0)

        pred_instances = InstanceData(priors=priors, valid_flags=valid_flags)
        gt_instances = InstanceData(bboxes=gt_bboxes, labels=gt_labels)
        num_level_anchors = [4]
        assign_result = region_assigner.assign(
            pred_instances, gt_instances, self.img_meta, self.featmap_sizes,
            num_level_anchors, self.anchor_scale, self.anchor_strides)

        expected_gt_inds = torch.LongTensor([0, 0, 0, 0])
        self.assertTrue(torch.all(assign_result.gt_inds == expected_gt_inds))

    def test_atss_assigner_with_empty_boxes(self):
        """Test corner case where a network might predict no boxes."""
        region_assigner = RegionAssigner(center_ratio=0.5)
        priors = torch.empty((0, 4))
        valid_flags = torch.BoolTensor([])
        gt_bboxes = torch.FloatTensor([
            [0, 0, 10, 9],
            [0, 10, 10, 19],
        ])
        gt_labels = torch.LongTensor([2, 3])

        pred_instances = InstanceData(priors=priors, valid_flags=valid_flags)
        gt_instances = InstanceData(bboxes=gt_bboxes, labels=gt_labels)
        num_level_anchors = [0]
        assign_result = region_assigner.assign(
            pred_instances, gt_instances, self.img_meta, self.featmap_sizes,
            num_level_anchors, self.anchor_scale, self.anchor_strides)
        self.assertEqual(len(assign_result.gt_inds), 0)
        self.assertTrue(tuple(assign_result.labels.shape) == (0, ))

    def test_atss_assigner_with_empty_boxes_and_gt(self):
        """Test corner case where a network might predict no boxes and no
        gt."""
        region_assigner = RegionAssigner(center_ratio=0.5)
        priors = torch.empty((0, 4))
        valid_flags = torch.BoolTensor([])
        gt_bboxes = torch.empty((0, 4))
        gt_labels = torch.empty(0)
        num_level_anchors = [0]

        pred_instances = InstanceData(priors=priors, valid_flags=valid_flags)
        gt_instances = InstanceData(bboxes=gt_bboxes, labels=gt_labels)
        assign_result = region_assigner.assign(
            pred_instances, gt_instances, self.img_meta, self.featmap_sizes,
            num_level_anchors, self.anchor_scale, self.anchor_strides)
        self.assertEqual(len(assign_result.gt_inds), 0)
