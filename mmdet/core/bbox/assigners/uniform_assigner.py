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
                 pos_iou_thr,
                 neg_iou_thr,
                 match_times=4,
                 iou_calculator=dict(type='BboxOverlaps2D')):
        self.match_times = match_times
        self.pos_iou_thr = pos_iou_thr
        self.neg_iou_thr = neg_iou_thr
        self.iou_calculator = build_iou_calculator(iou_calculator)

    def assign(self, bbox_pred, anchor, gt_bboxes, gt_labels, img_meta):
        num_gts, num_bboxes = gt_bboxes.size(0), bbox_pred.size(0)

        # 1. assign -1 by default
        assigned_gt_inds = bbox_pred.new_full((num_bboxes, ),
                                              -1,
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
        # cost_bbox = torch.cdist(
        #     box_xyxy_to_cxcywh(bbox_pred),
        #     box_xyxy_to_cxcywh(gt_bboxes), p=1)
        # cost_bbox_anchors = torch.cdist(
        #     box_xyxy_to_cxcywh(anchor),
        #     box_xyxy_to_cxcywh(gt_bboxes), p=1)

        # Final cost matrix
        # C = cost_bbox
        # C = C.view(bs, num_queries, -1).cpu()
        # C1 = cost_bbox_anchors
        # C1 = C1.view(bs, num_queries, -1).cpu()
