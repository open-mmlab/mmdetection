# Copyright (c) OpenMMLab. All rights reserved.
import torch

from ..builder import BBOX_ASSIGNERS
from ..iou_calculators import build_iou_calculator
from ..transforms import bbox_xyxy_to_cxcywh
from .assign_result import AssignResult
from .base_assigner import BaseAssigner


@BBOX_ASSIGNERS.register_module()
class UniformAssigner(BaseAssigner):
    """Uniform Matching between the anchors and gt boxes, which can achieve
    balance in positive anchors, and gt_bboxes_ignore was not considered for
    now.

    Args:
        pos_ignore_thr (float): the threshold to ignore positive anchors
        neg_ignore_thr (float): the threshold to ignore negative anchors
        match_times(int): Number of positive anchors for each gt box.
           Default 4.
        iou_calculator (dict): iou_calculator config
    """

    def __init__(self,
                 pos_ignore_thr,
                 neg_ignore_thr,
                 match_times=4,
                 iou_calculator=dict(type='BboxOverlaps2D')):
        self.match_times = match_times
        self.pos_ignore_thr = pos_ignore_thr
        self.neg_ignore_thr = neg_ignore_thr
        self.iou_calculator = build_iou_calculator(iou_calculator)

    def assign(self,
               bbox_pred,
               anchor,
               gt_bboxes,
               gt_bboxes_ignore=None,
               gt_labels=None):
        num_gts, num_bboxes = gt_bboxes.size(0), bbox_pred.size(0)

        # 1. assign -1 by default
        assigned_gt_inds = bbox_pred.new_full((num_bboxes, ),
                                              0,
                                              dtype=torch.long)
        assigned_labels = bbox_pred.new_full((num_bboxes, ),
                                             -1,
                                             dtype=torch.long)
        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            assign_result = AssignResult(
                num_gts, assigned_gt_inds, None, labels=assigned_labels)
            assign_result.set_extra_property(
                'pos_idx', bbox_pred.new_empty(0, dtype=torch.bool))
            assign_result.set_extra_property('pos_predicted_boxes',
                                             bbox_pred.new_empty((0, 4)))
            assign_result.set_extra_property('target_boxes',
                                             bbox_pred.new_empty((0, 4)))
            return assign_result

        # 2. Compute the L1 cost between boxes
        # Note that we use anchors and predict boxes both
        cost_bbox = torch.cdist(
            bbox_xyxy_to_cxcywh(bbox_pred),
            bbox_xyxy_to_cxcywh(gt_bboxes),
            p=1)
        cost_bbox_anchors = torch.cdist(
            bbox_xyxy_to_cxcywh(anchor), bbox_xyxy_to_cxcywh(gt_bboxes), p=1)

        # We found that topk function has different results in cpu and
        # cuda mode. In order to ensure consistency with the source code,
        # we also use cpu mode.
        # TODO: Check whether the performance of cpu and cuda are the same.
        C = cost_bbox.cpu()
        C1 = cost_bbox_anchors.cpu()

        # self.match_times x n
        index = torch.topk(
            C,  # c=b,n,x c[i]=n,x
            k=self.match_times,
            dim=0,
            largest=False)[1]

        # self.match_times x n
        index1 = torch.topk(C1, k=self.match_times, dim=0, largest=False)[1]
        # (self.match_times*2) x n
        indexes = torch.cat((index, index1),
                            dim=1).reshape(-1).to(bbox_pred.device)

        pred_overlaps = self.iou_calculator(bbox_pred, gt_bboxes)
        anchor_overlaps = self.iou_calculator(anchor, gt_bboxes)
        pred_max_overlaps, _ = pred_overlaps.max(dim=1)
        anchor_max_overlaps, _ = anchor_overlaps.max(dim=0)

        # 3. Compute the ignore indexes use gt_bboxes and predict boxes
        ignore_idx = pred_max_overlaps > self.neg_ignore_thr
        assigned_gt_inds[ignore_idx] = -1

        # 4. Compute the ignore indexes of positive sample use anchors
        # and predict boxes
        pos_gt_index = torch.arange(
            0, C1.size(1),
            device=bbox_pred.device).repeat(self.match_times * 2)
        pos_ious = anchor_overlaps[indexes, pos_gt_index]
        pos_ignore_idx = pos_ious < self.pos_ignore_thr

        pos_gt_index_with_ignore = pos_gt_index + 1
        pos_gt_index_with_ignore[pos_ignore_idx] = -1
        assigned_gt_inds[indexes] = pos_gt_index_with_ignore

        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), -1)
            pos_inds = torch.nonzero(
                assigned_gt_inds > 0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None

        assign_result = AssignResult(
            num_gts,
            assigned_gt_inds,
            anchor_max_overlaps,
            labels=assigned_labels)
        assign_result.set_extra_property('pos_idx', ~pos_ignore_idx)
        assign_result.set_extra_property('pos_predicted_boxes',
                                         bbox_pred[indexes])
        assign_result.set_extra_property('target_boxes',
                                         gt_bboxes[pos_gt_index])
        return assign_result
