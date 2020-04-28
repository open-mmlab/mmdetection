import torch

from ..builder import BBOX_ASSIGNERS
from ..iou_calculators import build_iou_calculator
from .assign_result import AssignResult
from .base_assigner import BaseAssigner


def scale_boxes(bboxes, scale):
    """Expand an array of boxes by a given scale.

    Args:
        bboxes (tensor): Shape (m, 4)
        scale (float): The scale factor of bboxes

    Returns:
        (tensor): Shape (m, 4). Scaled bboxes
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
    """Are points located in bboxes

    Args:
      points (tensor): Points, shape: [m,2].
      bboxes (tensor): Bounding boxes, shape: [n,4].

    Return:
      Flags indicating if points are located in bboxes, shape: [m, n].
    """
    assert points.size(1) == 2
    assert bboxes.size(1) == 4
    return (points[:, 0].unsqueeze(1) > bboxes[:, 0].unsqueeze(0)) & \
           (points[:, 0].unsqueeze(1) < bboxes[:, 2].unsqueeze(0)) & \
           (points[:, 1].unsqueeze(1) > bboxes[:, 1].unsqueeze(0)) & \
           (points[:, 1].unsqueeze(1) < bboxes[:, 3].unsqueeze(0))


def bboxes_area(bboxes):
    """Compute the area of an array of boxes."""
    assert bboxes.size(1) == 4
    w = (bboxes[:, 2] - bboxes[:, 0])
    h = (bboxes[:, 3] - bboxes[:, 1])
    areas = w * h

    return areas


@BBOX_ASSIGNERS.register_module
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

        This method assigns gts to every bbox (proposal/anchor), each bbox will
         be assigned with -1, or a semi-positive number. -1 means negative
         sample, semi-positive number is the index (0-based) of assigned gt.

        Args:
            bboxes (tensor): Bounding boxes to be assigned, shape(n, 4).
            gt_bboxes (tensor): Groundtruth boxes, shape (k, 4).
            gt_bboxes_ignore (tensor, optional): Ground truth bboxes that are
              labelled as `ignored`, e.g., crowd boxes in COCO.
            gt_labels (tensor, optional): Label of gt_bboxes, shape (num_gts,).

        Returns:
            :obj:`AssignResult`: The assign result.

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
            gt_areas = bboxes_area(gt_bboxes)
            # Rank all gt bbox areas. Smaller objects has larger priority
            _, sort_idx = gt_areas.sort(descending=True)
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
            assigned_labels = assigned_gt_ids.new_full((num_bboxes, ), -1)
            pos_inds = torch.nonzero(assigned_gt_ids > 0).squeeze()
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
                # When a pixel is both positive and shadowed, set it as shadow.
                override = (
                    assigned_labels[pixel_idx] == shadowed_pixel_labels[:, 1])
                assigned_labels[pixel_idx[override]] = -1
                assigned_gt_ids[pixel_idx[override]] = 0

        assign_result = AssignResult(
            num_gts, assigned_gt_ids, None, labels=assigned_labels)
        assign_result.set_extra_property('shadowed_labels',
                                         shadowed_pixel_labels)
        return assign_result

    def assign_one_hot_gt_indices(self,
                                  is_bbox_in_gt_core,
                                  is_bbox_in_gt_shadow,
                                  gt_priority=None):
        """Assign only one gt index to each prior box

        Smaller gts have higher priority

        Args:
            is_bbox_in_gt_core (tensor): shape [num_prior, num_gt].
              Bool tensor indicating the bbox center is in the core area of a
              gt (e.g. 0-0.2)
            is_bbox_in_gt_shadow (tensor): shape [num_prior, num_gt].
              Bool tensor indicating the bbox center is in the shadowed area
              of a gt (e.g. 0.2-0.5)
            gt_priority (tensor): shape [num_gt]. gt priorities.
              The gt with a higher priority is more likely to be
              assigned to the bbox when the bbox match with multiple gts

        Returns:
            assigned_gt_inds: Shape [num_prior]. The assigned gt index of each
              prior bbox (i.e. index from 1 to num_gts).
            shadowed_gt_inds: shadowed gt indices. It is a tensor of shape
              [num_ignore, 2] with first column being the shadowed prior bbox
              indices and the second column the shadowed gt indices
        """
        num_bboxes, num_gts = is_bbox_in_gt_core.shape

        if gt_priority is None:
            gt_priority = torch.arange(num_gts).to(is_bbox_in_gt_core.device)
            # the bigger, the more preferable to be assigned
        # the assigned inds are by default 0 (background)
        assigned_gt_inds = is_bbox_in_gt_core.new_full((num_bboxes, ),
                                                       0,
                                                       dtype=torch.long)
        inds_of_match = torch.any(is_bbox_in_gt_core, dim=1)
        # matched  bboxes (to any gt)
        inds_of_shadow = torch.any(is_bbox_in_gt_shadow, dim=1)
        # Ignored indices are assigned to be background. But the corresponding
        #   label is shadowed during loss calculation, which is done through
        #   shadowed_gt_inds
        assigned_gt_inds[inds_of_shadow] = 0
        shadowed_gt_inds = torch.nonzero(is_bbox_in_gt_shadow)
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

        # Whether a bbox match a gt,  bool tensor, shape [num_match, num_gt]
        matched_bbox_and_gt_correspondence = is_bbox_in_gt_core[inds_of_match]
        # The matched gt index of each positive bbox. Length >= num_match,
        #  since one bbox could match multiple gts
        matched_bbox_gt_inds =\
            torch.nonzero(matched_bbox_and_gt_correspondence)[:, 1]
        # Assign priority to each bbox-gt pair.
        pair_priority[is_bbox_in_gt_core] = gt_priority[matched_bbox_gt_inds]
        _, argmax_priority = pair_priority[inds_of_match].max(dim=1)
        # the maximum shape [num_match]
        # effective indices.
        assigned_gt_inds[inds_of_match] = argmax_priority + 1  # 1-based
        # Zero-out the assigned prior box to filter the shadowed gt indices
        is_bbox_in_gt_core[inds_of_match, argmax_priority] = 0
        # Concat the shadowed indices due to overlapping with that out side of
        #   effective scale. shape: [total_num_ignore, 2]
        shadowed_gt_inds = torch.cat(
            (shadowed_gt_inds, torch.nonzero(is_bbox_in_gt_core)), dim=0)
        # `is_bbox_in_gt_core` should be changed back to keep arguments intact.
        is_bbox_in_gt_core[inds_of_match, argmax_priority] = 1
        # 1-based shadowed gt indices, to be consistent with `assigned_gt_inds`
        shadowed_gt_inds[:, 1] += 1
        return assigned_gt_inds, shadowed_gt_inds
