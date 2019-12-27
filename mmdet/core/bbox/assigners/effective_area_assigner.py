import torch

from ..geometry import bbox_overlaps, scale_boxes, is_located_in, bboxes_area
from .assign_result import AssignResult
from .base_assigner import BaseAssigner


class EffectiveAreaAssigner(BaseAssigner):
    """Assign a corresponding gt bbox or background to each bbox.

    Each proposals will be assigned with `-1`, `0`, or a positive integer
    indicating the ground truth index.

    - -1: don't care
    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:
        pos_area_thr (float): threshold within which pixels are labelled as positive.
        neg_area_thr (float): threshold above which pixels are labelled as positive.
        min_pos_iof (float): minimum iof of a pixel with a gt to be labelled as positive
    """

    def __init__(self,
                 pos_area_thr,
                 neg_area_thr,
                 min_pos_iof=1e-2):
        self.pos_area_thr = pos_area_thr
        self.neg_area_thr = neg_area_thr
        self.min_pos_iof = min_pos_iof

    def assign(self, bboxes, gt_bboxes, gt_bboxes_ignore=None, gt_labels=None):
        """Assign gt to bboxes.

        This method assign a gt bbox to every bbox (proposal/anchor), each bbox
        will be assigned with -1, 0, or a positive number. -1 means don't care,
        0 means negative sample, positive number is the index (1-based) of
        assigned gt.
        The assignment is done in following steps, the order matters.

        1. assign every bbox to -1
        2. assign proposals whose iou with all gts < neg_iou_thr to 0
        3. for each bbox, if the iou with its nearest gt >= pos_iou_thr,
           assign it to that bbox
        4. for each gt bbox, assign its nearest proposals (may be more than
           one) to itself

        Args:
            bboxes (Tensor): Bounding boxes to be assigned, shape(n, 4).
            gt_bboxes (Tensor): Groundtruth boxes, shape (k, 4).
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO.
            gt_labels (Tensor, optional): Label of gt_bboxes, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.
        """
        if bboxes.shape[0] == 0 or gt_bboxes.shape[0] == 0:
            raise ValueError('No gt or bboxes')
        bboxes = bboxes[:, :4]

        ## constructing effective gt areas
        gt_eff = scale_boxes(gt_bboxes, self.pos_area_thr) # effective bboxes, i.e. center 0.2 part
        bbox_centers = (bboxes[:, 2:4] + bboxes[:, 0:2] + 1) / 2
        is_bbox_in_gt = is_located_in(bbox_centers, gt_bboxes)
        # the center points lie within the gt boxes

        bbox_and_gt_eff_overlaps = bbox_overlaps(bboxes, gt_eff, mode='iof')
        is_bbox_in_gt_eff = is_bbox_in_gt &\
                            (bbox_and_gt_eff_overlaps > self.min_pos_iof) # shape (n, k)
        # the center point of effective priors should be within the gt box

        ## constructing ignored gt areas
        gt_ignore = scale_boxes(gt_bboxes, self.neg_area_thr)
        is_bbox_in_gt_ignore = (bbox_overlaps(bboxes, gt_ignore, mode='iof') > self.min_pos_iof)
        is_bbox_in_gt_ignore &= (~is_bbox_in_gt_eff) # rule out center effective pixels


        gt_areas = bboxes_area(gt_bboxes)
        _, sort_idx = gt_areas.sort(descending=True)  # smaller instances can overlay larger ones
        assign_result = self.choose_from_multiple_assigns(is_bbox_in_gt_eff,
                                                          is_bbox_in_gt_ignore,
                                                          gt_labels,
                                                          gt_priority=sort_idx)
        return assign_result


    def choose_from_multiple_assigns(self, is_bbox_in_gt_eff,
                                     is_bbox_in_gt_ignore,
                                     gt_labels=None,
                                     gt_priority=None):
        """
        Assign the label of each prior box with regard to the rank of gt areas
        (smaller gt has higher priority)
        Args:
            is_bbox_in_gt_eff: shape [num_prior, num_gt]. bool tensor indicating the bbox
            center is in the effective area of a gt (e.g. 0-0.2)
            is_bbox_in_gt_ignore: shape [num_prior, num_gt]. bool tensor indicating the bbox
            center is in the ignored area of a gt (e.g. 0.2-0.5)
            gt_labels: shape [num_gt]. gt labels (0-81 for COCO)
            gt_priority: shape [num_gt]. gt priorities. The gt with a higher priority is more
                likely to be assigned to the bbox when the bbox match with multiple gts
        Returns:
            :obj:`AssignResult`: The assign result.
        """
        num_bboxes, num_gts = is_bbox_in_gt_eff.size(0), is_bbox_in_gt_eff.size(1)
        if gt_priority is None:
            gt_priority = torch.arange(num_gts).to(is_bbox_in_gt_eff.device)
            # the bigger, the more preferable to be assigned

        assigned_gt_inds = is_bbox_in_gt_eff.new_full((num_bboxes,),
                                                      0,
                                                      dtype=torch.long)
        inds_of_match = torch.any(is_bbox_in_gt_eff, dim=1)  # matched  bboxes (to any gt)
        inds_of_ignore = torch.any(is_bbox_in_gt_ignore, dim=1)  # ignored indices

        assigned_gt_inds[inds_of_ignore] = -1
        if is_bbox_in_gt_eff.sum() == 0: # No gt match
            return AssignResult(num_gts, assigned_gt_inds, None, labels=None)

        bbox_priority = is_bbox_in_gt_eff.new_full((num_bboxes, num_gts),
                                                   -1,
                                                   dtype=torch.long)

        # Each bbox could match with multiple gts. The following codes deal with this
        matched_bbox_and_gt_correspondence = is_bbox_in_gt_eff[inds_of_match]  # shape [nmatch, k]
        matched_bbox_gt_inds = torch.nonzero(matched_bbox_and_gt_correspondence)[:, 1]
        # the matched gt index of each positive bbox. shape [nmatch]
        bbox_priority[is_bbox_in_gt_eff] = gt_priority[matched_bbox_gt_inds]
        _, argmax_priority = bbox_priority[inds_of_match].max(dim=1) # the maximum shape [nmatch]
        #effective indices
        assigned_gt_inds[inds_of_match] = argmax_priority + 1

        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_zeros((num_bboxes, ))
            pos_inds = torch.nonzero(assigned_gt_inds > 0).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None

        return AssignResult(
            num_gts, assigned_gt_inds, None, labels=assigned_labels)
