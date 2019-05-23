import torch

from .max_iou_assigner import MaxIoUAssigner
from ..geometry import bbox_overlaps


class ApproxMaxIoUAssigner(MaxIoUAssigner):
    """Assign a corresponding gt bbox or background to each bbox.

    Each proposals will be assigned with `-1`, `0`, or a positive integer
    indicating the ground truth index.

    - -1: don't care
    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:
        pos_iou_thr (float): IoU threshold for positive bboxes.
        neg_iou_thr (float or tuple): IoU threshold for negative bboxes.
        min_pos_iou (float): Minimum iou for a bbox to be considered as a
            positive bbox. Positive samples can have smaller IoU than
            pos_iou_thr due to the 4th step (assign max IoU sample to each gt).
        gt_max_assign_all (bool): Whether to assign all bboxes with the same
            highest overlap with some gt to that gt.
        ignore_iof_thr (float): IoF threshold for ignoring bboxes (if
            `gt_bboxes_ignore` is specified). Negative values mean not
            ignoring any bboxes.
        ignore_wrt_candidates (bool): Whether to compute the iof between
            `bboxes` and `gt_bboxes_ignore`, or the contrary.
    """

    def __init__(self,
                 pos_iou_thr,
                 neg_iou_thr,
                 min_pos_iou=.0,
                 gt_max_assign_all=True,
                 ignore_iof_thr=-1,
                 ignore_wrt_candidates=True):
        self.pos_iou_thr = pos_iou_thr
        self.neg_iou_thr = neg_iou_thr
        self.min_pos_iou = min_pos_iou
        self.gt_max_assign_all = gt_max_assign_all
        self.ignore_iof_thr = ignore_iof_thr
        self.ignore_wrt_candidates = ignore_wrt_candidates

    def assign(self,
               approxs,
               squares,
               approxs_per_octave,
               gt_bboxes,
               gt_bboxes_ignore=None,
               gt_labels=None):
        """Assign gt to approxs.

        This method assign a gt bbox to each group of approxs (bboxes),
        each group of approxs is represent by a base approx (bbox) and
        will be assigned with -1, 0, or a positive number.
        -1 means don't care, 0 means negative sample,
        positive number is the index (1-based) of assigned gt.
        The assignment is done in following steps, the order matters.

        1. assign every bbox to -1
        2. use the max IoU of each group of approxs to assign
        2. assign proposals whose iou with all gts < neg_iou_thr to 0
        3. for each bbox, if the iou with its nearest gt >= pos_iou_thr,
           assign it to that bbox
        4. for each gt bbox, assign its nearest proposals (may be more than
           one) to itself

        Args:
            approxs (Tensor): Bounding boxes to be assigned,
        shape(approxs_per_octave*n, 4).
            squares (Tensor): Base Bounding boxes to be assigned,
        shape(n, 4).
            approxs_per_octave (int): number of approxs per octave
            gt_bboxes (Tensor): Groundtruth boxes, shape (k, 4).
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO.
            gt_labels (Tensor, optional): Label of gt_bboxes, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.
        """

        if squares.shape[0] == 0 or gt_bboxes.shape[0] == 0:
            raise ValueError('No gt or approxs')
        num_squares = squares.size(0)
        num_gts = gt_bboxes.size(0)
        # re-organize anchors by approxs_per_octave x num_squares
        approxs = torch.transpose(
            approxs.view(num_squares, approxs_per_octave, 4), 0,
            1).contiguous().view(-1, 4)
        all_overlaps = bbox_overlaps(approxs, gt_bboxes)

        overlaps, _ = all_overlaps.view(approxs_per_octave, num_squares,
                                        num_gts).max(dim=0)
        overlaps = torch.transpose(overlaps, 0, 1)

        bboxes = squares[:, :4]

        if (self.ignore_iof_thr > 0) and (gt_bboxes_ignore is not None) and (
                gt_bboxes_ignore.numel() > 0):
            if self.ignore_wrt_candidates:
                ignore_overlaps = bbox_overlaps(bboxes,
                                                gt_bboxes_ignore,
                                                mode='iof')
                ignore_max_overlaps, _ = ignore_overlaps.max(dim=1)
            else:
                ignore_overlaps = bbox_overlaps(gt_bboxes_ignore,
                                                bboxes,
                                                mode='iof')
                ignore_max_overlaps, _ = ignore_overlaps.max(dim=0)
            overlaps[:, ignore_max_overlaps > self.ignore_iof_thr] = -1

        assign_result = self.assign_wrt_overlaps(overlaps, gt_labels)
        return assign_result
