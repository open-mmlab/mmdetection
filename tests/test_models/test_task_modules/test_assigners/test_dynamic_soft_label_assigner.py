# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
from mmengine.structures import InstanceData
from mmengine.testing import assert_allclose

from mmdet.models.task_modules.assigners import DynamicSoftLabelAssigner
from mmdet.structures.bbox import HorizontalBoxes


class TestDynamicSoftLabelAssigner(TestCase):

    def test_assign(self):
        assigner = DynamicSoftLabelAssigner(
            soft_center_radius=3.0, topk=1, iou_weight=3.0)
        pred_instances = InstanceData(
            bboxes=torch.Tensor([[23, 23, 43, 43], [4, 5, 6, 7]]),
            scores=torch.FloatTensor([[0.2], [0.8]]),
            priors=torch.Tensor([[30, 30, 8, 8], [4, 5, 6, 7]]))
        gt_instances = InstanceData(
            bboxes=torch.Tensor([[23, 23, 43, 43]]),
            labels=torch.LongTensor([0]))
        assign_result = assigner.assign(
            pred_instances=pred_instances, gt_instances=gt_instances)

        expected_gt_inds = torch.LongTensor([1, 0])
        assert_allclose(assign_result.gt_inds, expected_gt_inds)

    def test_assign_with_no_valid_bboxes(self):
        assigner = DynamicSoftLabelAssigner(
            soft_center_radius=3.0, topk=1, iou_weight=3.0)
        pred_instances = InstanceData(
            bboxes=torch.Tensor([[123, 123, 143, 143], [114, 151, 161, 171]]),
            scores=torch.FloatTensor([[0.2], [0.8]]),
            priors=torch.Tensor([[30, 30, 8, 8], [55, 55, 8, 8]]))
        gt_instances = InstanceData(
            bboxes=torch.Tensor([[0, 0, 1, 1]]), labels=torch.LongTensor([0]))
        assign_result = assigner.assign(
            pred_instances=pred_instances, gt_instances=gt_instances)

        expected_gt_inds = torch.LongTensor([0, 0])
        assert_allclose(assign_result.gt_inds, expected_gt_inds)

    def test_assign_with_empty_gt(self):
        assigner = DynamicSoftLabelAssigner(
            soft_center_radius=3.0, topk=1, iou_weight=3.0)
        pred_instances = InstanceData(
            bboxes=torch.Tensor([[[30, 40, 50, 60]], [[4, 5, 6, 7]]]),
            scores=torch.FloatTensor([[0.2], [0.8]]),
            priors=torch.Tensor([[0, 12, 23, 34], [4, 5, 6, 7]]))
        gt_instances = InstanceData(
            bboxes=torch.empty(0, 4), labels=torch.empty(0))

        assign_result = assigner.assign(
            pred_instances=pred_instances, gt_instances=gt_instances)
        expected_gt_inds = torch.LongTensor([0, 0])
        assert_allclose(assign_result.gt_inds, expected_gt_inds)

    def test_box_type_input(self):
        assigner = DynamicSoftLabelAssigner(
            soft_center_radius=3.0, topk=1, iou_weight=3.0)
        pred_instances = InstanceData(
            bboxes=torch.Tensor([[23, 23, 43, 43], [4, 5, 6, 7]]),
            scores=torch.FloatTensor([[0.2], [0.8]]),
            priors=torch.Tensor([[30, 30, 8, 8], [4, 5, 6, 7]]))
        gt_instances = InstanceData(
            bboxes=HorizontalBoxes(torch.Tensor([[23, 23, 43, 43]])),
            labels=torch.LongTensor([0]))
        assign_result = assigner.assign(
            pred_instances=pred_instances, gt_instances=gt_instances)

        expected_gt_inds = torch.LongTensor([1, 0])
        assert_allclose(assign_result.gt_inds, expected_gt_inds)
