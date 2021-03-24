import torch

from ..builder import BBOX_ASSIGNERS
from ..iou_calculators import build_iou_calculator
from .assign_result import AssignResult
from .base_assigner import BaseAssigner


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


@BBOX_ASSIGNERS.register_module()
class UniformAssigner(BaseAssigner):

    def __init__(self,
                 pos_ignore_thresh,
                 neg_ignore_thresh,
                 match_times=4,
                 iou_calculator=dict(type='BboxOverlaps2D')):
        self.match_times = match_times
        self.pos_ignore_thresh = pos_ignore_thresh
        self.neg_ignore_thresh = neg_ignore_thresh
        self.iou_calculator = build_iou_calculator(iou_calculator)

    def assign(self, bbox_pred, anchor, gt_bboxes, gt_labels, img_meta):
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
            return AssignResult(
                num_gts, assigned_gt_inds, None, labels=assigned_labels)

        # Compute the L1 cost between boxes
        # Note that we use anchors and predict boxes both
        cost_bbox = torch.cdist(
            box_xyxy_to_cxcywh(bbox_pred), box_xyxy_to_cxcywh(gt_bboxes), p=1)
        cost_bbox_anchors = torch.cdist(
            box_xyxy_to_cxcywh(anchor), box_xyxy_to_cxcywh(gt_bboxes), p=1)

        # Final cost matrix
        C = cost_bbox
        C1 = cost_bbox_anchors

        # self.match_times x n
        indices = torch.topk(
            C,  # c=b,n,x c[i]=n,x
            k=self.match_times,
            dim=0,
            largest=False)[1]

        # self.match_times x n
        indices1 = torch.topk(C1, k=self.match_times, dim=0, largest=False)[1]
        # (self.match_times*2) x n
        indeices = torch.cat((indices, indices1), dim=0)

        pred_overlaps = self.iou_calculator(bbox_pred, gt_bboxes)
        anchor_overlaps = self.iou_calculator(anchor, gt_bboxes)

        pred_max_overlaps, pred_argmax_overlaps = pred_overlaps.max(dim=1)
        anchor_max_overlaps, _ = anchor_overlaps.max(dim=0)
        ignore_idx = pred_max_overlaps > self.neg_ignore_thresh
        assigned_gt_inds[ignore_idx] = -1

        pos_ious = torch.gather(anchor_overlaps, 0, indeices)
        pos_gt_index = torch.arange(
            0, pos_ious.size(1), device=pos_ious.device).expand_as(pos_ious)
        pos_ious = pos_ious.view(-1)
        indeices = indeices.view(-1)
        pos_gt_index = pos_gt_index.reshape(-1)
        pos_idx = pos_ious >= self.pos_ignore_thresh
        assigned_gt_inds[indeices[pos_idx]] = pos_gt_index[pos_idx] + 1
        assigned_gt_inds[indeices[~pos_idx]] = -1

        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), -1)
            pos_inds = torch.nonzero(
                assigned_gt_inds > 0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None

        return AssignResult(
            num_gts,
            assigned_gt_inds,
            anchor_max_overlaps,
            labels=assigned_labels)
