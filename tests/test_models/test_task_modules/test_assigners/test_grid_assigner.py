# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
from mmengine.structures import InstanceData
from mmengine.testing import assert_allclose

from mmdet.models.task_modules.assigners import GridAssigner


class TestGridAssigner(TestCase):

    def test_assign(self):
        assigner = GridAssigner(pos_iou_thr=0.5, neg_iou_thr=0.3)
        pred_instances = InstanceData(
            priors=torch.Tensor([[23, 23, 43, 43], [4, 5, 6, 7]]),
            responsible_flags=torch.BoolTensor([1, 1]))
        gt_instances = InstanceData(
            bboxes=torch.Tensor([[23, 23, 43, 43]]),
            labels=torch.LongTensor([0]))
        assign_result = assigner.assign(
            pred_instances=pred_instances, gt_instances=gt_instances)
        expected_gt_inds = torch.LongTensor([1, 0])
        assert_allclose(assign_result.gt_inds, expected_gt_inds)

        # invalid neg_iou_thr
        with self.assertRaises(AssertionError):
            assigner = GridAssigner(
                pos_iou_thr=0.5, neg_iou_thr=[0.3, 0.1, 0.4])
            assigner.assign(
                pred_instances=pred_instances, gt_instances=gt_instances)

        # multi-neg_iou_thr
        assigner = GridAssigner(pos_iou_thr=0.5, neg_iou_thr=(0.1, 0.3))
        assign_result = assigner.assign(
            pred_instances=pred_instances, gt_instances=gt_instances)
        expected_gt_inds = torch.LongTensor([1, -1])
        assert_allclose(assign_result.gt_inds, expected_gt_inds)

        # gt_max_assign_all=False
        assigner = GridAssigner(
            pos_iou_thr=0.5, neg_iou_thr=0.3, gt_max_assign_all=False)
        assign_result = assigner.assign(
            pred_instances=pred_instances, gt_instances=gt_instances)
        expected_gt_inds = torch.LongTensor([1, 0])
        assert_allclose(assign_result.gt_inds, expected_gt_inds)

        # large min_pos_iou
        assigner = GridAssigner(
            pos_iou_thr=0.5, neg_iou_thr=0.3, min_pos_iou=1)
        assign_result = assigner.assign(
            pred_instances=pred_instances, gt_instances=gt_instances)
        expected_gt_inds = torch.LongTensor([1, 0])
        assert_allclose(assign_result.gt_inds, expected_gt_inds)

    def test_assign_with_empty_gt(self):
        assigner = GridAssigner(pos_iou_thr=0.5, neg_iou_thr=0.3)
        pred_instances = InstanceData(
            priors=torch.Tensor([[0, 12, 23, 34], [4, 5, 6, 7]]),
            responsible_flags=torch.BoolTensor([1, 1]))
        gt_instances = InstanceData(
            bboxes=torch.empty(0, 4), labels=torch.empty(0))

        assign_result = assigner.assign(
            pred_instances=pred_instances, gt_instances=gt_instances)
        expected_gt_inds = torch.LongTensor([0, 0])
        assert_allclose(assign_result.gt_inds, expected_gt_inds)

    def test_assign_with_empty_priors(self):
        assigner = GridAssigner(pos_iou_thr=0.5, neg_iou_thr=0.3)
        pred_instances = InstanceData(
            priors=torch.Tensor(torch.empty(0, 4)),
            responsible_flags=torch.empty(0))
        gt_instances = InstanceData(
            bboxes=torch.Tensor([[23, 23, 43, 43]]),
            labels=torch.LongTensor([0]))

        assign_result = assigner.assign(
            pred_instances=pred_instances, gt_instances=gt_instances)
        expected_gt_inds = torch.LongTensor([])
        assert_allclose(assign_result.gt_inds, expected_gt_inds)
