# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch
from mmengine.structures import InstanceData

from mmdet.registry import TASK_UTILS
from .assign_result import AssignResult
from .max_iou_assigner import MaxIoUAssigner


@TASK_UTILS.register_module()
class MultiInstanceAssigner(MaxIoUAssigner):
    """Assign a corresponding gt bbox or background to each bbox.

    Args:
        num_instance (int): How many bboxes are predicted by each proposal box.
    """

    def __init__(self, num_instance: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.num_instance = num_instance

    def assign(self,
               pred_instances: InstanceData,
               gt_instances: InstanceData,
               gt_instances_ignore: Optional[InstanceData] = None,
               **kwargs) -> AssignResult:
        gt_bboxes = gt_instances.bboxes
        priors = pred_instances.priors
        # Set the FG label to 1 and add ignored annotations
        gt_labels = gt_instances.labels + 1
        gt_bboxes_ignore = gt_instances_ignore.bboxes
        gt_labels_ignore = gt_instances_ignore.labels

        assign_on_cpu = True if (self.gpu_assign_thr > 0) and (
            gt_bboxes.shape[0] > self.gpu_assign_thr) else False
        # compute overlap and assign gt on CPU when number of GT is large
        if assign_on_cpu:
            device = priors.device
            priors = priors.cpu()
            gt_bboxes = gt_bboxes.cpu()
            gt_labels = gt_labels.cpu()
            if gt_bboxes_ignore is not None:
                gt_bboxes_ignore = gt_bboxes_ignore.cpu()

        all_bboxes = torch.cat([gt_bboxes, gt_bboxes_ignore], dim=0)
        all_labels = torch.cat([gt_labels, gt_labels_ignore], dim=0)
        all_priors = torch.cat([priors, all_bboxes], dim=0)

        overlaps_normal = self.iou_calculator(
            all_priors, all_bboxes, mode='iou')
        overlaps_ignore = self.iou_calculator(
            all_priors, all_bboxes, mode='iof')
        gt_ignore_mask = all_labels.eq(-1).repeat(all_priors.shape[0], 1)
        overlaps_normal = overlaps_normal * ~gt_ignore_mask
        overlaps_ignore = overlaps_ignore * gt_ignore_mask

        overlaps_normal, overlaps_normal_indices = overlaps_normal.sort(
            descending=True, dim=1)
        overlaps_ignore, overlaps_ignore_indices = overlaps_ignore.sort(
            descending=True, dim=1)

        # select the roi with the higher score
        max_overlaps_normal = overlaps_normal[:, :self.num_instance].flatten()
        gt_assignment_normal = overlaps_normal_indices[:, :self.
                                                       num_instance].flatten()
        max_overlaps_ignore = overlaps_ignore[:, :self.num_instance].flatten()
        gt_assignment_ignore = overlaps_ignore_indices[:, :self.
                                                       num_instance].flatten()

        # ignore or not
        ignore_assign_mask = (max_overlaps_normal < self.pos_iou_thr) * (
            max_overlaps_ignore > max_overlaps_normal)
        overlaps = (max_overlaps_normal * ~ignore_assign_mask) + (
            max_overlaps_ignore * ignore_assign_mask)
        gt_assignment = (gt_assignment_normal * ~ignore_assign_mask) + (
            gt_assignment_ignore * ignore_assign_mask)

        assigned_labels = all_labels[gt_assignment]
        fg_mask = (overlaps >= self.pos_iou_thr) * (assigned_labels != -1)
        bg_mask = (overlaps < self.neg_iou_thr) * (overlaps >= 0)
        assigned_labels[fg_mask] = 1
        assigned_labels[bg_mask] = 0

        overlaps = overlaps.reshape(-1, self.num_instance)
        gt_assignment = gt_assignment.reshape(-1, self.num_instance)
        assigned_labels = assigned_labels.reshape(-1, self.num_instance)

        assign_result = AssignResult(
            num_gts=all_bboxes.size(0),
            gt_inds=gt_assignment,
            max_overlaps=overlaps,
            labels=assigned_labels)

        if assign_on_cpu:
            assign_result.gt_inds = assign_result.gt_inds.to(device)
            assign_result.max_overlaps = assign_result.max_overlaps.to(device)
            if assign_result.labels is not None:
                assign_result.labels = assign_result.labels.to(device)
        return assign_result
