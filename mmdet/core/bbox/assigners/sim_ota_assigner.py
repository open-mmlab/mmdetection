import torch
import torch.nn.functional as F
from mmdet.core import bbox_overlaps

from ..builder import BBOX_ASSIGNERS
from ..match_costs import build_match_cost
from ..transforms import bbox_cxcywh_to_xyxy
from .assign_result import AssignResult
from .base_assigner import BaseAssigner


@BBOX_ASSIGNERS.register_module()
class SimOTAAssigner(BaseAssigner):
    """Computes matching between predictions and ground truth.

    Args:
        cls_weight (int | float, optional): The scale factor for classification
            cost. Default 1.0.
        bbox_weight (int | float, optional): The scale factor for regression
            L1 cost. Default 1.0.
        iou_weight (int | float, optional): The scale factor for regression
            iou cost. Default 1.0.
        iou_calculator (dict | optional): The config for the iou calculation.
            Default type `BboxOverlaps2D`.
        iou_mode (str | optional): "iou" (intersection over union), "iof"
                (intersection over foreground), or "giou" (generalized
                intersection over union). Default "giou".
    """

    def __init__(self, num_classes=80):
        self.num_classes = num_classes


    def assign(self, pred_scores, priors, decoded_bboxes, gt_bboxes,
               gt_labels,gt_bboxes_ignore=None, eps=1e-7):
        """

        Args:
            pred_scores:
            priors:
            gt_bboxes:
            gt_labels:
            img_meta:
            gt_bboxes_ignore:
            eps:

        Returns:

        """
        try:
            assign_results = self._assign(pred_scores, priors, decoded_bboxes, gt_bboxes,
                gt_labels, gt_bboxes_ignore, eps, device=decoded_bboxes.device)
        except RuntimeError:
            print(
                "OOM RuntimeError is raised due to the huge memory cost during label assignment. \
                   CPU mode is applied in this batch. If you want to avoid this issue, \
                   try to reduce the batch size or image size."
            )
            torch.cuda.empty_cache()
            assign_results = self._assign(pred_scores, priors, decoded_bboxes, gt_bboxes,
                gt_labels, gt_bboxes_ignore, eps, device='cpu')

        return assign_results

    def _assign(self, pred_scores, priors, decoded_bboxes, gt_bboxes,
               gt_labels, gt_bboxes_ignore=None, eps=1e-7, device='cuda'):
        INF = 100000000

        num_priors = priors.size(0)
        num_gt = gt_bboxes.size(0)

        # assign 0 by default
        assigned_gt_inds = priors.new_full((num_priors, ),
                                             0,
                                             dtype=torch.long)

        if num_gt == 0 or num_priors == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = priors.new_zeros((num_priors,))
            if num_gt == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = priors.new_full((num_priors,),
                                                     -1,
                                                     dtype=torch.long)
            return AssignResult(
                num_gt, assigned_gt_inds, None, labels=assigned_labels)

        valid_mask, is_in_boxes_and_center = self.get_in_gt_and_in_center_info(priors, gt_bboxes)

        valid_decoded_bbox = decoded_bboxes[valid_mask]
        valid_pred_scores = pred_scores[valid_mask]
        valid_priors = priors[valid_mask]
        num_valid = valid_decoded_bbox.size(0)

        # TODO: use match cost
        pair_wise_ious = bbox_overlaps(valid_decoded_bbox, gt_bboxes)
        iou_cost = -torch.log(pair_wise_ious + 1e-8)  # [num_valid, num_gt]

        gt_onehot_label = (
            F.one_hot(gt_labels.to(torch.int64), self.num_classes).float()
                .unsqueeze(0).repeat(num_valid, 1, 1)
        )

        valid_pred_scores = valid_pred_scores.unsqueeze(1).repeat(1, num_gt, 1)
        cls_cost = F.binary_cross_entropy(
            valid_pred_scores.sqrt_(), gt_onehot_label, reduction="none"
        ).sum(-1)  # [num_valid, num_gt]

        cost_matrix = cls_cost + 3.0 * iou_cost + INF * (~is_in_boxes_and_center)

        num_fg, gt_matched_classes, \
        pred_ious_this_matching, matched_gt_inds = self.dynamic_k_matching(cost_matrix, pair_wise_ious, gt_labels, num_gt, valid_mask)

        return gt_matched_classes, valid_mask, pred_ious_this_matching, matched_gt_inds, num_fg

    def get_in_gt_and_in_center_info(self, priors, gt_bboxes):
        center_radius = 2.5
        num_priors = priors.size(0)
        num_gt = gt_bboxes.size(0)

        repeated_x = priors[:, 0].unsqueeze(1).repeat(1, num_gt)
        repeated_y = priors[:, 1].unsqueeze(1).repeat(1, num_gt)
        repeated_stride_x = priors[:, 2].unsqueeze(1).repeat(1, num_gt)
        repeated_stride_y = priors[:, 3].unsqueeze(1).repeat(1, num_gt)

        # is prior centers in gt bboxes
        # [n_prior, n_gt]
        l_ = repeated_x - gt_bboxes[:, 0]
        t_ = repeated_y - gt_bboxes[:, 1]
        r_ = gt_bboxes[:, 2] - repeated_x
        b_ = gt_bboxes[:, 3] - repeated_y

        deltas = torch.stack([l_, t_, r_, b_], dim=1)
        is_in_gts = deltas.min(dim=1).values > 0
        is_in_gts_all = is_in_gts.sum(dim=1) > 0

        # is prior centers in gt centers
        gt_cxs = (gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2.0
        gt_cys = (gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2.0
        ct_box_l = gt_cxs - center_radius * repeated_stride_x
        ct_box_t = gt_cys - center_radius * repeated_stride_y
        ct_box_r = gt_cxs + center_radius * repeated_stride_x
        ct_box_b = gt_cys + center_radius * repeated_stride_y

        cl_ = repeated_x - ct_box_l
        ct_ = repeated_y - ct_box_t
        cr_ = ct_box_r - repeated_x
        cb_ = ct_box_b - repeated_y

        ct_deltas = torch.stack([cl_, ct_, cr_, cb_], dim=1)
        is_in_cts = ct_deltas.min(dim=1).values > 0
        is_in_cts_all = is_in_cts.sum(dim=1) > 0

        # in boxes or in centers
        is_in_gts_or_centers = is_in_gts_all | is_in_cts_all  # [num_priors]

        # TODO: fg outside
        # both in boxes and centers
        is_in_boxes_and_center = (
                is_in_gts[is_in_gts_or_centers, :] & is_in_cts[is_in_gts_or_centers, :]
        )  # shape [num_fg, num_gt]

        return is_in_gts_or_centers, is_in_boxes_and_center

    def dynamic_k_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
        # Dynamic K
        # ---------------------------------------------------------------
        matching_matrix = torch.zeros_like(cost)

        ious_in_boxes_matrix = pair_wise_ious  # [num_valid, num_gts]
        n_candidate_k = 10
        topk_ious, _ = torch.topk(ious_in_boxes_matrix, n_candidate_k, dim=0)
        dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=1)  # [num_gts]
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[:, gt_idx], k=dynamic_ks[gt_idx].item(), largest=False
            )
            matching_matrix[:, gt_idx][pos_idx] = 1.0

        del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(1)
        if (anchor_matching_gt > 1).sum() > 0:
            cost_min, cost_argmin = torch.min(cost[anchor_matching_gt > 1, :], dim=1)
            matching_matrix[anchor_matching_gt > 1, :] *= 0.0
            matching_matrix[anchor_matching_gt > 1, cost_argmin] = 1.0
        fg_mask_inboxes = matching_matrix.sum(1) > 0.0
        num_fg = fg_mask_inboxes.sum().item()

        fg_mask[fg_mask.clone()] = fg_mask_inboxes

        matched_gt_inds = matching_matrix[fg_mask_inboxes, :].argmax(1)
        gt_matched_classes = gt_classes[matched_gt_inds]

        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(1)[fg_mask_inboxes]
        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds
