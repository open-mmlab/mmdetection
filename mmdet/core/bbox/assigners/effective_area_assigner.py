import torch

from ..builder import BBOX_ASSIGNERS
from ..iou_calculators import build_iou_calculator
from .assign_result import AssignResult
from .base_assigner import BaseAssigner


def scale_boxes(bboxes, scale):
    """Expand an array of boxes by a given scale.
        Args:
            bboxes (Tensor): shape (m, 4)
            scale (float): the scale factor of bboxes

        Returns:
            (Tensor): shape (m, 4) scaled bboxes
        """
    w_half = (bboxes[:, 2] - bboxes[:, 0]) * .5
    h_half = (bboxes[:, 3] - bboxes[:, 1]) * .5
    x_c = (bboxes[:, 2] + bboxes[:, 0]) * .5
    y_c = (bboxes[:, 3] + bboxes[:, 1]) * .5

    w_half *= scale
    h_half *= scale

    boxes_exp = torch.zeros_like(bboxes)
    boxes_exp[:, 0] = x_c - w_half
    boxes_exp[:, 2] = x_c + w_half
    boxes_exp[:, 1] = y_c - h_half
    boxes_exp[:, 3] = y_c + h_half
    return boxes_exp


def is_located_in(points, bboxes, is_aligned=False):
    """ is center a locates in box b
    Then we compute the area of intersect between box_a and box_b.
    Args:
      points: (tensor) bounding boxes, Shape: [m,2].
      bboxes: (tensor)  bounding boxes, Shape: [n,4].
       If is_aligned is ``True``, then m mush be equal to n
    Return:
      (tensor) intersection area, Shape: [m, n]. If is_aligned ``True``,
       then shape = [m]
    """
    if not is_aligned:
        return (points[:, 0].unsqueeze(1) > bboxes[:, 0].unsqueeze(0)) & \
               (points[:, 0].unsqueeze(1) < bboxes[:, 2].unsqueeze(0)) & \
               (points[:, 1].unsqueeze(1) > bboxes[:, 1].unsqueeze(0)) & \
               (points[:, 1].unsqueeze(1) < bboxes[:, 3].unsqueeze(0))
    else:
        return (points[:, 0] > bboxes[:, 0]) & \
               (points[:, 0] < bboxes[:, 2]) & \
               (points[:, 1] > bboxes[:, 1]) & \
               (points[:, 1] < bboxes[:, 3])


def bboxes_area(bboxes):
    """Compute the area of an array of boxes."""
    w = (bboxes[:, 2] - bboxes[:, 0])
    h = (bboxes[:, 3] - bboxes[:, 1])
    areas = w * h

    return areas


@BBOX_ASSIGNERS.register_module
class EffectiveAreaAssigner(BaseAssigner):
    """Assign a corresponding gt bbox or background to each bbox.

    Each proposals will be assigned with `-1`, `0`, or a positive integer
    indicating the ground truth index.

    - -1: don't care
    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:
        pos_area_thr (float): threshold within which pixels are
          labelled as positive.
        neg_area_thr (float): threshold above which pixels are
          labelled as positive.
        min_pos_iof (float): minimum iof of a pixel with a gt to be
          labelled as positive
        ignore_gt_area_thr (float): threshold within which the pixels
        are ignored when the gt is labelled as ignored
    """

    def __init__(self,
                 pos_scale,
                 neg_scale,
                 min_pos_iof=1e-2,
                 ignore_gt_scale=0.5,
                 iou_calculator=dict(type='BboxOverlaps2D')):
        self.pos_scale = pos_scale
        self.neg_scale = neg_scale
        self.min_pos_iof = min_pos_iof
        self.ignore_gt_scale = ignore_gt_scale
        self.iou_calculator = build_iou_calculator(iou_calculator)

    def assign(self, bboxes, gt_bboxes, gt_bboxes_ignore=None, gt_labels=None):
        """Assign gt to bboxes.

        This method assign a gt bbox to every bbox (proposal/anchor), each bbox
        will be assigned with -1, 0, or a positive number. -1 means don't care,
        0 means negative sample, positive number is the index (1-based) of
        assigned gt.

        Args:
            bboxes (Tensor): Bounding boxes to be assigned, shape(n, 4).
            gt_bboxes (Tensor): Groundtruth boxes, shape (k, 4).
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO.
            gt_labels (Tensor, optional): Label of gt_bboxes, shape (num_gt, ).

        Returns:
            :obj:`AssignResult`: The assign result.
        """
        if bboxes.shape[0] == 0 or gt_bboxes.shape[0] == 0:
            raise ValueError('No gt or bboxes')
        bboxes = bboxes[:, :4]

        # constructing effective gt areas
        gt_eff = scale_boxes(gt_bboxes, self.pos_scale)
        # effective bboxes, i.e. the center 0.2 part
        bbox_centers = (bboxes[:, 2:4] + bboxes[:, 0:2]) / 2
        is_bbox_in_gt = is_located_in(bbox_centers, gt_bboxes)
        # the center points lie within the gt boxes

        # Only calculate bbox and gt_eff IoF. This enables small prior bboxes
        #   to match large gts
        bbox_and_gt_eff_overlaps = self.iou_calculator(
            bboxes, gt_eff, mode='iof')
        is_bbox_in_gt_eff = is_bbox_in_gt & (
            bbox_and_gt_eff_overlaps > self.min_pos_iof)
        # shape (n, k)
        # the center point of effective priors should be within the gt box

        # constructing ignored gt areas
        gt_ignore = scale_boxes(gt_bboxes, self.neg_scale)
        is_bbox_in_gt_ignore = (
            self.iou_calculator(bboxes, gt_ignore, mode='iof') >
            self.min_pos_iof)
        is_bbox_in_gt_ignore &= (~is_bbox_in_gt_eff)
        # rule out center effective pixels

        gt_areas = bboxes_area(gt_bboxes)
        _, sort_idx = gt_areas.sort(descending=True)
        # rank all gt bbox areas so that smaller instances
        #   can overlay larger ones

        assigned_gt_inds, ignored_gt_inds = self.assign_one_hot_gt_indices(
            is_bbox_in_gt_eff, is_bbox_in_gt_ignore, gt_priority=sort_idx)

        # gt bboxes that are either crowded or to be ignored
        if gt_bboxes_ignore is not None and gt_bboxes_ignore.numel() > 0:
            gt_bboxes_ignore = scale_boxes(
                gt_bboxes_ignore, scale=self.ignore_gt_scale)
            is_bbox_in_ignored_gts = is_located_in(bbox_centers,
                                                   gt_bboxes_ignore)
            is_bbox_in_ignored_gts = is_bbox_in_ignored_gts.any(dim=1)
            assigned_gt_inds[is_bbox_in_ignored_gts] = -1

        num_bboxes, num_gts = is_bbox_in_gt_eff.shape
        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_zeros((num_bboxes, ))
            pos_inds = torch.nonzero(assigned_gt_inds > 0).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1]
            # convert from ignored gt indices to ignored gt label
            ignored_gt_labels = ignored_gt_inds.clone()
            if ignored_gt_inds.numel() > 0:
                idx, gt_idx = ignored_gt_inds[:, 0], ignored_gt_inds[:, 1]
                assert (assigned_gt_inds[idx] != gt_idx).all(), \
                    'Some pixels are dually assigned to ignore and gt!'
                ignored_gt_labels[:, 1] = gt_labels[gt_idx - 1]
                # Positive labels can override ignored labels, e.g. when
                #   a small horse stands at the ignored region of a large one.
                override = (assigned_labels[idx] == ignored_gt_labels[:, 1])
                ignored_gt_labels = ignored_gt_labels[~override]
        else:
            assigned_labels = None
            ignored_gt_labels = None

        return AssignResult(
            num_gts,
            assigned_gt_inds,
            None,
            labels=assigned_labels,
            ignored_labels=ignored_gt_labels)

    def assign_one_hot_gt_indices(self,
                                  is_bbox_in_gt_eff,
                                  is_bbox_in_gt_ignore,
                                  gt_priority=None):
        """Assign only one gt index to each prior box
        (smaller gt has higher priority)

        Args:
            is_bbox_in_gt_eff: shape [num_prior, num_gt].
              bool tensor indicating the bbox center is in
              the effective area of a gt (e.g. 0-0.2)
            is_bbox_in_gt_ignore: shape [num_prior, num_gt].
              bool tensor indicating the bbox
            center is in the ignored area of a gt (e.g. 0.2-0.5)
            gt_labels: shape [num_gt]. gt labels (0-80 for COCO)
            gt_priority: shape [num_gt]. gt priorities.
              The gt with a higher priority is more likely to be
              assigned to the bbox when the bbox match with multiple gts

        Returns:
            :obj:`AssignResult`: The assign result.
            ignored_gt_inds: ignored gt indices. It is a tensor of shape
                [num_ignore, 2] with first column recording the ignored feature
                map indices and the second column the ignored gt indices
        """
        num_bboxes, num_gts = is_bbox_in_gt_eff.shape

        if gt_priority is None:
            gt_priority = torch.arange(num_gts).to(is_bbox_in_gt_eff.device)
            # the bigger, the more preferable to be assigned
        # the assigned inds are by default 0 (background)
        assigned_gt_inds = is_bbox_in_gt_eff.new_full((num_bboxes, ),
                                                      0,
                                                      dtype=torch.long)
        inds_of_match = torch.any(is_bbox_in_gt_eff, dim=1)
        # matched  bboxes (to any gt)
        inds_of_ignore = torch.any(is_bbox_in_gt_ignore, dim=1)
        # Ignored indices are assigned to be background. But the corresponding
        #   label is ignored during loss calculation, which is done through
        #   ignored_gt_inds
        assigned_gt_inds[inds_of_ignore] = 0
        ignored_gt_inds = torch.nonzero(is_bbox_in_gt_ignore)
        if is_bbox_in_gt_eff.sum() == 0:  # No gt match
            return assigned_gt_inds, ignored_gt_inds

        # The priority of each prior box and gt pair. If one prior box is
        #  matched bo multiple gts. Only the pair with the highest priority
        #  is saved
        pair_priority = is_bbox_in_gt_eff.new_full((num_bboxes, num_gts),
                                                   -1,
                                                   dtype=torch.long)

        # Each bbox could match with multiple gts.
        # The following codes deal with this situation

        # Whether a bbox match a gt,  bool tensor, shape [num_match, num_gt]
        matched_bbox_and_gt_correspondence = is_bbox_in_gt_eff[inds_of_match]
        # The matched gt index of each positive bbox. Length >= num_match,
        #  since one bbox could match multiple gts
        matched_bbox_gt_inds =\
            torch.nonzero(matched_bbox_and_gt_correspondence)[:, 1]
        # Assign priority to each bbox-gt pair.
        pair_priority[is_bbox_in_gt_eff] = gt_priority[matched_bbox_gt_inds]
        _, argmax_priority = pair_priority[inds_of_match].max(dim=1)
        # the maximum shape [num_match]
        # effective indices. Note that positive assignment can overwrite
        # negative or ignored ones
        assigned_gt_inds[inds_of_match] = argmax_priority + 1  # 1-based
        # Zero-out the assigned pixels to filter the ignored gt indices
        is_bbox_in_gt_eff[inds_of_match, argmax_priority] = 0
        # Concat the ignored indices due to overlapping with that out side of
        #   effective scale. shape: [total_num_ignore, 2]
        ignored_gt_inds = torch.cat(
            (ignored_gt_inds, torch.nonzero(is_bbox_in_gt_eff)), dim=0)
        ignored_gt_inds[:, 1] += 1  # 1-based. For consistency issue
        return assigned_gt_inds, ignored_gt_inds
