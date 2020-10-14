import torch
from scipy.optimize import linear_sum_assignment

from ..builder import BBOX_ASSIGNERS
from ..transforms import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh
from .assign_result import AssignResult
from .base_assigner import BaseAssigner


@BBOX_ASSIGNERS.register_module()
class HungarianMatchAssigner(BaseAssigner):
    """Computes one-to-one matching between predictions and ground truth.

    This class computes an assignment between the targets and the predictions
    based on the costs. The costs are weighted sum of three components:
    classfication cost, regression L1 cost and regression giou cost. The
    targets don't include the no_object, so generally there are more
    predictions than targets. After the one-to-one matching, the un-matched
    are treated as backgrounds. Thus each query prediction will be assigned
    with `0` or a positive integer indicating the ground truth index:

    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:
        cls_wei (int | float, optional): The scale factor for classification
            cost. Default 1.0.
        bbox_wei (int | float, optional): The scale factor for regression
            L1 cost. Default 1.0.
        giou_wei (int | float, optional): The scale factor for regression
            giou cost. Default 1.0.
    """

    def __init__(self, cls_wei=1., bbox_wei=1., giou_wei=1.):
        self.cls_wei = cls_wei
        self.bbox_wei = bbox_wei
        self.giou_wei = giou_wei

    def assign(self,
               bbox_pred,
               cls_pred,
               gt_bboxes,
               gt_labels,
               img_meta,
               gt_bboxes_ignore=None,
               eps=1e-7):
        """Computes one-to-one matching based on the weighted costs.

        This method assign each query prediction to a ground truth or
        background. The `assigned_gt_inds` with -1 means don't care,
        0 means negative sample, and positive number is the index (1-based)
        of assigned gt.
        The assignment is done in following steps, the order matters.

        1. assign every prediction to -1
        2. compute the weighted costs
        3. do Hungarian matching on CPU based on the costs
        4. assign all to 0 (background) first, then for each matched pair
           between predictions and gts, treat this prediction as foreground
           and assign the corresponding gt index (plus 1) to it.

        Args:
            bbox_pred (Tensor): Predicted boxes with normalized coordinates
                (cx, cy, w, h), which are all in range [0, 1]. Shape
                [num_query, 4].
            cls_pred (Tensor): Predicted classification logits, shape
                [num_query, num_class].
            gt_bboxes (Tensor): Ground truth boxes with unnormalized
                coordinates (x1, y1, x2, y2). Shape [num_gt, 4].
            gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).
            img_meta (dict): Meta information for current image.
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`. Default None.
            eps (int | float, optional): A value added to the denominator for
                numerical stability. Default 1e-7.

        Returns:
            :obj:`AssignResult`: The assigned result.
        """
        assert gt_bboxes_ignore is None, \
            'Only case when gt_bboxes_ignore is None is supported.'
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

        # 2. compute the weighted costs
        # classification cost.
        # Following the official DETR repo, contrary to the loss that
        # NLL is used, we approximate it in 1 - cls_score[gt_label].
        # The 1 is a constant that doesn't change the matching,
        # so it can be ommitted.
        cls_score = cls_pred.softmax(-1)
        cls_cost = -cls_score[:, gt_labels]  # [num_query, num_gt]

        # regression L1 cost
        img_h, img_w, _ = img_meta['img_shape']
        factor = torch.Tensor([img_w, img_h, img_w,
                               img_h]).unsqueeze(0).to(gt_bboxes.device)
        gt_bboxes_normalized = gt_bboxes / factor
        bbox_cost = torch.cdist(
            bbox_pred, bbox_xyxy_to_cxcywh(gt_bboxes_normalized),
            p=1)  # [num_query, num_gt]

        # regression giou cost
        bboxes = bbox_cxcywh_to_xyxy(bbox_pred) * factor
        lt = torch.max(bboxes[:, None, :2], gt_bboxes[None, :, :2])
        rb = torch.min(bboxes[:, None, 2:], gt_bboxes[None, :, 2:])
        wh = (rb - lt).clamp(min=0)
        overlap = wh[..., 0] * wh[..., 1]
        # calculate ious
        area1 = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        area2 = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (
            gt_bboxes[:, 3] - gt_bboxes[:, 1])
        union = area1[:, None] + area2[None, :] - overlap
        union = union.clamp(min=eps)
        ious = overlap / union
        # calculate enclose_area
        enclose_lt = torch.min(bboxes[:, None, :2], gt_bboxes[None, :, :2])
        enclose_rb = torch.max(bboxes[:, None, 2:], gt_bboxes[None, :, 2:])
        enclose_wh = (enclose_rb - enclose_lt).clamp(min=0)
        enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
        enclose_area = enclose_area.clamp(min=eps)
        # calculate giou
        gious = ious - (enclose_area - union) / enclose_area
        # The 1 is a constant that doesn't change the matching, so ommitted.
        giou_cost = -gious

        # weighted sum of above three costs
        cost = self.cls_wei * cls_cost + self.bbox_wei * bbox_cost
        cost = cost + self.giou_wei * giou_cost

        # 3. do Hungarian matching on CPU using linear_sum_assignment
        cost = cost.detach().cpu()
        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        matched_row_inds = torch.from_numpy(matched_row_inds).to(
            bbox_pred.device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(
            bbox_pred.device)

        # 4. assign backgrounds and foregrounds
        # assign all indices to backgrounds first
        assigned_gt_inds[:] = 0
        # assign foregrounds based on matching results
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]
        return AssignResult(
            num_gts, assigned_gt_inds, None, labels=assigned_labels)
