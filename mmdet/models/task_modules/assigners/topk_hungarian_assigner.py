# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmengine.structures import BaseDataElement
from scipy.optimize import linear_sum_assignment

from mmdet.registry import TASK_UTILS
from .assign_result import AssignResult
from .task_aligned_assigner import TaskAlignedAssigner


@TASK_UTILS.register_module()
class TopkHungarianAssigner(TaskAlignedAssigner):

    def __init__(self,
                 *args,
                 cls_cost=dict(type='FocalLossCost', weight=2.0),
                 reg_cost=dict(type='BBoxL1Cost', weight=5.0),
                 iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0),
                 **kwargs):
        super(TopkHungarianAssigner, self).__init__(*args, **kwargs)

        self.cls_cost = TASK_UTILS.build(cls_cost)
        self.reg_cost = TASK_UTILS.build(reg_cost)
        self.iou_cost = TASK_UTILS.build(iou_cost)

    def assign(self,
               pred_scores,
               decode_bboxes,
               gt_bboxes,
               gt_labels,
               img_meta,
               alpha=1,
               beta=6,
               **kwargs):
        pred_scores = pred_scores.detach()
        decode_bboxes = decode_bboxes.detach()
        temp_overlaps = self.iou_calculator(decode_bboxes, gt_bboxes).detach()
        bbox_scores = pred_scores[:, gt_labels].detach()
        alignment_metrics = bbox_scores**alpha * temp_overlaps**beta

        pred_instances = BaseDataElement()
        gt_instances = BaseDataElement()

        pred_instances.bboxes = decode_bboxes
        gt_instances.bboxes = gt_bboxes

        pred_instances.scores = pred_scores
        gt_instances.labels = gt_labels

        reg_cost = self.reg_cost(pred_instances, gt_instances, img_meta)
        iou_cost = self.iou_cost(pred_instances, gt_instances, img_meta)
        cls_cost = self.cls_cost(pred_instances, gt_instances, img_meta)
        all_cost = cls_cost + reg_cost + iou_cost

        num_gt, num_bboxes = gt_bboxes.size(0), pred_scores.size(0)
        if num_gt > 0:
            # assign 0 by default
            assigned_gt_inds = pred_scores.new_full((num_bboxes, ),
                                                    0,
                                                    dtype=torch.long)
            select_cost = all_cost
            # num anchor * (num_gt * topk)
            topk = min(self.topk, int(len(select_cost) / num_gt))
            # num_anchors * (num_gt * topk)
            repeat_select_cost = select_cost[...,
                                             None].repeat(1, 1, topk).view(
                                                 select_cost.size(0), -1)
            # anchor index and gt index
            matched_row_inds, matched_col_inds = linear_sum_assignment(
                repeat_select_cost.detach().cpu().numpy())
            matched_row_inds = torch.from_numpy(matched_row_inds).to(
                pred_scores.device)
            matched_col_inds = torch.from_numpy(matched_col_inds).to(
                pred_scores.device)

            match_gt_ids = matched_col_inds // topk
            candidate_idxs = matched_row_inds

            assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), -1)

            if candidate_idxs.numel() > 0:
                assigned_labels[candidate_idxs] = gt_labels[match_gt_ids]
            else:
                assigned_labels = None

            assigned_gt_inds[candidate_idxs] = match_gt_ids + 1

            overlaps = self.iou_calculator(
                decode_bboxes[candidate_idxs],
                gt_bboxes[match_gt_ids],
                is_aligned=True).detach()

            temp_pos_alignment_metrics = alignment_metrics[candidate_idxs]
            pos_alignment_metrics = torch.gather(temp_pos_alignment_metrics, 1,
                                                 match_gt_ids[:,
                                                              None]).view(-1)
            assign_result = AssignResult(
                num_gt, assigned_gt_inds, overlaps, labels=assigned_labels)

            assign_result.assign_metrics = pos_alignment_metrics
            return assign_result
        else:

            assigned_gt_inds = pred_scores.new_full((num_bboxes, ),
                                                    -1,
                                                    dtype=torch.long)

            assigned_labels = pred_scores.new_full((num_bboxes, ),
                                                   -1,
                                                   dtype=torch.long)

            assigned_gt_inds[:] = 0
            return AssignResult(
                0, assigned_gt_inds, None, labels=assigned_labels)
