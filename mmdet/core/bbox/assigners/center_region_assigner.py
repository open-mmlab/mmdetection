# Copyright (c) OpenMMLab. All rights reserved.
import torch

from ..builder import BBOX_ASSIGNERS
from ..iou_calculators import build_iou_calculator
from .assign_result import AssignResult
from .base_assigner import BaseAssigner


def scale_boxes(bboxes, scale):
    """Expand an array of boxes by a given scale.

    Args:
        bboxes (Tensor): Shape (m, 4)
        scale (float): The scale factor of bboxes

    Returns:
        (Tensor): Shape (m, 4). Scaled bboxes
    """
    assert bboxes.size(1) == 4
    w_half = (bboxes[:, 2] - bboxes[:, 0]) * .5
    h_half = (bboxes[:, 3] - bboxes[:, 1]) * .5
    x_c = (bboxes[:, 2] + bboxes[:, 0]) * .5
    y_c = (bboxes[:, 3] + bboxes[:, 1]) * .5

    w_half *= scale
    h_half *= scale

    boxes_scaled = torch.zeros_like(bboxes)
    boxes_scaled[:, 0] = x_c - w_half
    boxes_scaled[:, 2] = x_c + w_half
    boxes_scaled[:, 1] = y_c - h_half
    boxes_scaled[:, 3] = y_c + h_half
    return boxes_scaled


def is_located_in(points, bboxes):
    """Are points located in bboxes.

    Args:
      points (Tensor): Points, shape: (m, 2).
      bboxes (Tensor): Bounding boxes, shape: (n, 4).

    Return:
      Tensor: Flags indicating if points are located in bboxes, shape: (m, n).
    """
    assert points.size(1) == 2
    assert bboxes.size(1) == 4
    return (points[:, 0].unsqueeze(1) > bboxes[:, 0].unsqueeze(0)) & \
           (points[:, 0].unsqueeze(1) < bboxes[:, 2].unsqueeze(0)) & \
           (points[:, 1].unsqueeze(1) > bboxes[:, 1].unsqueeze(0)) & \
           (points[:, 1].unsqueeze(1) < bboxes[:, 3].unsqueeze(0))


def bboxes_area(bboxes):
    """Compute the area of an array of bboxes.

    Args:
        bboxes (Tensor): The coordinates ox bboxes. Shape: (m, 4)

    Returns:
        Tensor: Area of the bboxes. Shape: (m, )
    """
    assert bboxes.size(1) == 4
    w = (bboxes[:, 2] - bboxes[:, 0])
    h = (bboxes[:, 3] - bboxes[:, 1])
    areas = w * h
    return areas


@BBOX_ASSIGNERS.register_module()
class CenterRegionAssigner(BaseAssigner):
    """Assign pixels at the center region of a bbox as positive.

    Each proposals will be assigned with `-1`, `0`, or a positive integer
    indicating the ground truth index.
    - -1: negative samples
    - semi-positive numbers: positive sample, index (0-based) of assigned gt

    Args:
        pos_scale (float): Threshold within which pixels are
          labelled as positive.
        neg_scale (float): Threshold above which pixels are
          labelled as positive.
        min_pos_iof (float): Minimum iof of a pixel with a gt to be
          labelled as positive. Default: 1e-2
        ignore_gt_scale (float): Threshold within which the pixels
          are ignored when the gt is labelled as shadowed. Default: 0.5
        foreground_dominate (bool): If True, the bbox will be assigned as
          positive when a gt's kernel region overlaps with another's shadowed
          (ignored) region, otherwise it is set as ignored. Default to False.
    """

    def __init__(self,
                 pos_scale,
                 neg_scale,
                 min_pos_iof=1e-2,
                 ignore_gt_scale=0.5,
                 foreground_dominate=False,
                 iou_calculator=dict(type='BboxOverlaps2D')):
        self.pos_scale = pos_scale
        self.neg_scale = neg_scale
        self.min_pos_iof = min_pos_iof
        self.ignore_gt_scale = ignore_gt_scale
        self.foreground_dominate = foreground_dominate
        self.iou_calculator = build_iou_calculator(iou_calculator)

    def get_gt_priorities(self, gt_bboxes):
        """Get gt priorities according to their areas.

        Smaller gt has higher priority.

        Args:
            gt_bboxes (Tensor): Ground truth boxes, shape (k, 4).

        Returns:
            Tensor: The priority of gts so that gts with larger priority is \
              more likely to be assigned. Shape (k, )
        """
        gt_areas = bboxes_area(gt_bboxes)
        # Rank all gt bbox areas. Smaller objects has larger priority
        _, sort_idx = gt_areas.sort(descending=True)
        sort_idx = sort_idx.argsort()
        return sort_idx

    def assign(self, bboxes, gt_bboxes, gt_bboxes_ignore=None, gt_labels=None):
        """Assign gt to bboxes.

        This method assigns gts to every bbox (proposal/anchor), each bbox \
        will be assigned with -1, or a semi-positive number. -1 means \
        negative sample, semi-positive number is the index (0-based) of \
        assigned gt.

        Args:
            bboxes (Tensor): Bounding boxes to be assigned, shape(n, 4).
            gt_bboxes (Tensor): Groundtruth boxes, shape (k, 4).
            gt_bboxes_ignore (tensor, optional): Ground truth bboxes that are
              labelled as `ignored`, e.g., crowd boxes in COCO.
            gt_labels (tensor, optional): Label of gt_bboxes, shape (num_gts,).

        Returns:
            :obj:`AssignResult`: The assigned result. Note that \
              shadowed_labels of shape (N, 2) is also added as an \
              `assign_result` attribute. `shadowed_labels` is a tensor \
              composed of N pairs of anchor_ind, class_label], where N \
              is the number of anchors that lie in the outer region of a \
              gt, anchor_ind is the shadowed anchor index and class_label \
              is the shadowed class label.

        Example:
            >>> self = CenterRegionAssigner(0.2, 0.2)
            >>> bboxes = torch.Tensor([[0, 0, 10, 10], [10, 10, 20, 20]])
            >>> gt_bboxes = torch.Tensor([[0, 0, 10, 10]])
            >>> assign_result = self.assign(bboxes, gt_bboxes)
            >>> expected_gt_inds = torch.LongTensor([1, 0])
            >>> assert torch.all(assign_result.gt_inds == expected_gt_inds)
        """
        # There are in total 5 steps in the pixel assignment
        # 1. Find core (the center region, say inner 0.2)
        #     and shadow (the relatively ourter part, say inner 0.2-0.5)
        #     regions of every gt.
        # 2. Find all prior bboxes that lie in gt_core and gt_shadow regions
        # 3. Assign prior bboxes in gt_core with a one-hot id of the gt in
        #      the image.
        #    3.1. For overlapping objects, the prior bboxes in gt_core is
        #           assigned with the object with smallest area
        # 4. Assign prior bboxes with class label according to its gt id.
        #    4.1. Assign -1 to prior bboxes lying in shadowed gts
        #    4.2. Assign positive prior boxes with the corresponding label
        # 5. Find pixels lying in the shadow of an object and assign them with
        #      background label, but set the loss weight of its corresponding
        #      gt to zero.
        assert bboxes.size(1) == 4, 'bboxes must have size of 4'
        # 1. Find core positive and shadow region of every gt
        gt_core = scale_boxes(gt_bboxes, self.pos_scale)
        gt_shadow = scale_boxes(gt_bboxes, self.neg_scale)

        # 2. Find prior bboxes that lie in gt_core and gt_shadow regions
        bbox_centers = (bboxes[:, 2:4] + bboxes[:, 0:2]) / 2
        # The center points lie within the gt boxes
        is_bbox_in_gt = is_located_in(bbox_centers, gt_bboxes)
        # Only calculate bbox and gt_core IoF. This enables small prior bboxes
        #   to match large gts
        bbox_and_gt_core_overlaps = self.iou_calculator(
            bboxes, gt_core, mode='iof')
        # The center point of effective priors should be within the gt box
        is_bbox_in_gt_core = is_bbox_in_gt & (
            bbox_and_gt_core_overlaps > self.min_pos_iof)  # shape (n, k)

        is_bbox_in_gt_shadow = (
            self.iou_calculator(bboxes, gt_shadow, mode='iof') >
            self.min_pos_iof)
        # Rule out center effective positive pixels
        is_bbox_in_gt_shadow &= (~is_bbox_in_gt_core)

        num_gts, num_bboxes = gt_bboxes.size(0), bboxes.size(0)
        if num_gts == 0 or num_bboxes == 0:
            # If no gts exist, assign all pixels to negative
            assigned_gt_ids = \
                is_bbox_in_gt_core.new_zeros((num_bboxes,),
                                             dtype=torch.long)
            pixels_in_gt_shadow = assigned_gt_ids.new_empty((0, 2))
        else:
            # Step 3: assign a one-hot gt id to each pixel, and smaller objects
            #    have high priority to assign the pixel.
            sort_idx = self.get_gt_priorities(gt_bboxes)
            assigned_gt_ids, pixels_in_gt_shadow = \
                self.assign_one_hot_gt_indices(is_bbox_in_gt_core,
                                               is_bbox_in_gt_shadow,
                                               gt_priority=sort_idx)

        if gt_bboxes_ignore is not None and gt_bboxes_ignore.numel() > 0:
            # No ground truth or boxes, return empty assignment
            gt_bboxes_ignore = scale_boxes(
                gt_bboxes_ignore, scale=self.ignore_gt_scale)
            is_bbox_in_ignored_gts = is_located_in(bbox_centers,
                                                   gt_bboxes_ignore)
            is_bbox_in_ignored_gts = is_bbox_in_ignored_gts.any(dim=1)
            assigned_gt_ids[is_bbox_in_ignored_gts] = -1

        # 4. Assign prior bboxes with class label according to its gt id.
        assigned_labels = None
        shadowed_pixel_labels = None
        if gt_labels is not None:
            # Default assigned label is the background (-1)
            assigned_labels = assigned_gt_ids.new_full((num_bboxes, ), -1)
            pos_inds = torch.nonzero(
                assigned_gt_ids > 0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[assigned_gt_ids[pos_inds]
                                                      - 1]
            # 5. Find pixels lying in the shadow of an object
            shadowed_pixel_labels = pixels_in_gt_shadow.clone()
            if pixels_in_gt_shadow.numel() > 0:
                pixel_idx, gt_idx =\
                    pixels_in_gt_shadow[:, 0], pixels_in_gt_shadow[:, 1]
                assert (assigned_gt_ids[pixel_idx] != gt_idx).all(), \
                    'Some pixels are dually assigned to ignore and gt!'
                shadowed_pixel_labels[:, 1] = gt_labels[gt_idx - 1]
                override = (
                    assigned_labels[pixel_idx] == shadowed_pixel_labels[:, 1])
                if self.foreground_dominate:
                    # When a pixel is both positive and shadowed, set it as pos
                    shadowed_pixel_labels = shadowed_pixel_labels[~override]
                else:
                    # When a pixel is both pos and shadowed, set it as shadowed
                    assigned_labels[pixel_idx[override]] = -1
                    assigned_gt_ids[pixel_idx[override]] = 0

        assign_result = AssignResult(
            num_gts, assigned_gt_ids, None, labels=assigned_labels)
        # Add shadowed_labels as assign_result property. Shape: (num_shadow, 2)
        assign_result.set_extra_property('shadowed_labels',
                                         shadowed_pixel_labels)
        return assign_result

    def assign_one_hot_gt_indices(self,
                                  is_bbox_in_gt_core,
                                  is_bbox_in_gt_shadow,
                                  gt_priority=None):
        """Assign only one gt index to each prior box.

        Gts with large gt_priority are more likely to be assigned.

        Args:
            is_bbox_in_gt_core (Tensor): Bool tensor indicating the bbox center
              is in the core area of a gt (e.g. 0-0.2).
              Shape: (num_prior, num_gt).
            is_bbox_in_gt_shadow (Tensor): Bool tensor indicating the bbox
              center is in the shadowed area of a gt (e.g. 0.2-0.5).
              Shape: (num_prior, num_gt).
            gt_priority (Tensor): Priorities of gts. The gt with a higher
              priority is more likely to be assigned to the bbox when the bbox
              match with multiple gts. Shape: (num_gt, ).

        Returns:
            tuple: Returns (assigned_gt_inds, shadowed_gt_inds).

                - assigned_gt_inds: The assigned gt index of each prior bbox \
                    (i.e. index from 1 to num_gts). Shape: (num_prior, ).
                - shadowed_gt_inds: shadowed gt indices. It is a tensor of \
                    shape (num_ignore, 2) with first column being the \
                    shadowed prior bbox indices and the second column the \
                    shadowed gt indices (1-based).
        """
        num_bboxes, num_gts = is_bbox_in_gt_core.shape

        if gt_priority is None:
            gt_priority = torch.arange(
                num_gts, device=is_bbox_in_gt_core.device)
        assert gt_priority.size(0) == num_gts
        # The bigger gt_priority, the more preferable to be assigned
        # The assigned inds are by default 0 (background)
        assigned_gt_inds = is_bbox_in_gt_core.new_zeros((num_bboxes, ),
                                                        dtype=torch.long)
        # Shadowed bboxes are assigned to be background. But the corresponding
        #   label is ignored during loss calculation, which is done through
        #   shadowed_gt_inds
        shadowed_gt_inds = torch.nonzero(is_bbox_in_gt_shadow, as_tuple=False)
        if is_bbox_in_gt_core.sum() == 0:  # No gt match
            shadowed_gt_inds[:, 1] += 1  # 1-based. For consistency issue
            return assigned_gt_inds, shadowed_gt_inds

        # The priority of each prior box and gt pair. If one prior box is
        #  matched bo multiple gts. Only the pair with the highest priority
        #  is saved
        pair_priority = is_bbox_in_gt_core.new_full((num_bboxes, num_gts),
                                                    -1,
                                                    dtype=torch.long)

        # Each bbox could match with multiple gts.
        # The following codes deal with this situation
        # Matched  bboxes (to any gt). Shape: (num_pos_anchor, )
        inds_of_match = torch.any(is_bbox_in_gt_core, dim=1)
        # The matched gt index of each positive bbox. Length >= num_pos_anchor
        #   , since one bbox could match multiple gts
        matched_bbox_gt_inds = torch.nonzero(
            is_bbox_in_gt_core, as_tuple=False)[:, 1]
        # Assign priority to each bbox-gt pair.
        pair_priority[is_bbox_in_gt_core] = gt_priority[matched_bbox_gt_inds]
        _, argmax_priority = pair_priority[inds_of_match].max(dim=1)
        assigned_gt_inds[inds_of_match] = argmax_priority + 1  # 1-based
        # Zero-out the assigned anchor box to filter the shadowed gt indices
        is_bbox_in_gt_core[inds_of_match, argmax_priority] = 0
        # Concat the shadowed indices due to overlapping with that out side of
        #   effective scale. shape: (total_num_ignore, 2)
        shadowed_gt_inds = torch.cat(
            (shadowed_gt_inds, torch.nonzero(
                is_bbox_in_gt_core, as_tuple=False)),
            dim=0)
        # `is_bbox_in_gt_core` should be changed back to keep arguments intact.
        is_bbox_in_gt_core[inds_of_match, argmax_priority] = 1
        # 1-based shadowed gt indices, to be consistent with `assigned_gt_inds`
        if shadowed_gt_inds.numel() > 0:
            shadowed_gt_inds[:, 1] += 1
        return assigned_gt_inds, shadowed_gt_inds
