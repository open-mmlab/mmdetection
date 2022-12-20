# Copyright (c) OpenMMLab. All rights reserved.
import torch

from ..builder import BBOX_ASSIGNERS
from ..iou_calculators import build_iou_calculator
from .ascend_assign_result import AscendAssignResult
from .base_assigner import BaseAssigner
from ....utils import set_index


@BBOX_ASSIGNERS.register_module()
class AscendMaxIoUAssigner(BaseAssigner):
    """Assign a corresponding gt bbox or background to each bbox.

    Each proposals will be assigned with `-1`, or a semi-positive integer
    indicating the ground truth index.

    - -1: negative sample, no assigned gt
    - semi-positive integer: positive sample, index (0-based) of assigned gt

    Args:
        pos_iou_thr (float): IoU threshold for positive bboxes.
        neg_iou_thr (float or tuple): IoU threshold for negative bboxes.
        min_pos_iou (float): Minimum iou for a bbox to be considered as a
            positive bbox. Positive samples can have smaller IoU than
            pos_iou_thr due to the 4th step (assign max IoU sample to each gt).
            `min_pos_iou` is set to avoid assigning bboxes that have extremely
            small iou with GT as positive samples. It brings about 0.3 mAP
            improvements in 1x schedule but does not affect the performance of
            3x schedule. More comparisons can be found in
            `PR #7464 <https://github.com/open-mmlab/mmdetection/pull/7464>`_.
        gt_max_assign_all (bool): Whether to assign all bboxes with the same
            highest overlap with some gt to that gt.
        ignore_iof_thr (float): IoF threshold for ignoring bboxes (if
            `gt_bboxes_ignore` is specified). Negative values mean not
            ignoring any bboxes.
        ignore_wrt_candidates (bool): Whether to compute the iof between
            `bboxes` and `gt_bboxes_ignore`, or the contrary.
        match_low_quality (bool): Whether to allow low quality matches. This is
            usually allowed for RPN and single stage detectors, but not allowed
            in the second stage. Details are demonstrated in Step 4.
        gpu_assign_thr (int): The upper bound of the number of GT for GPU
            assign. When the number of gt is above this threshold, will assign
            on CPU device. Negative values mean not assign on CPU.
    """

    def __init__(self,
                 pos_iou_thr,
                 neg_iou_thr,
                 min_pos_iou=.0,
                 gt_max_assign_all=True,
                 ignore_iof_thr=-1,
                 ignore_wrt_candidates=True,
                 match_low_quality=True,
                 gpu_assign_thr=-1,
                 iou_calculator=dict(type='BboxOverlaps2D')):
        self.pos_iou_thr = pos_iou_thr
        self.neg_iou_thr = neg_iou_thr
        self.min_pos_iou = min_pos_iou
        self.gt_max_assign_all = gt_max_assign_all
        self.ignore_iof_thr = ignore_iof_thr
        self.ignore_wrt_candidates = ignore_wrt_candidates
        self.gpu_assign_thr = gpu_assign_thr
        self.match_low_quality = match_low_quality
        self.iou_calculator = build_iou_calculator(iou_calculator)

    def assign(self,
               concat_bboxes,
               concat_gt_bboxes,
               concat_gt_bboxes_ignore=None,
               concat_gt_labels=None,
               concat_bboxes_ignore_mask=None,
               concat_num_gts=None):
        """Assign gt to bboxes.
        Args:
            concat_bboxes (Tensor): Bounding boxes to be assigned, shape(b, n, 4).
            concat_gt_bboxes (Tensor): Groundtruth boxes, shape (b, k, 4).
            concat_gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO.
            concat_gt_labels (Tensor, optional): Label of gt_bboxes, shape (b, k, ).
            concat_bboxes_ignore_mask: (b, n)
            concat_num_gts:(b, )
        Returns:
            :obj:`AssignResult`: The assign result.
        """
        concat_overlaps = self.iou_calculator(concat_gt_bboxes, concat_bboxes)
        concat_overlaps = set_index(concat_overlaps, concat_bboxes_ignore_mask.unsqueeze(1).float(), -1, neg=True)
        if self.ignore_iof_thr > 0 and concat_gt_bboxes_ignore is not None:
            if self.ignore_wrt_candidates:
                concat_ignore_overlaps = self.iou_calculator(concat_bboxes, concat_gt_bboxes_ignore, mode='iof')
                concat_ignore_overlaps = set_index(concat_ignore_overlaps, concat_bboxes_ignore_mask, -1)
                concat_ignore_max_overlaps, _ = concat_ignore_overlaps.max(dim=2)
            else:
                concat_ignore_overlaps = self.iou_calculator(concat_gt_bboxes_ignore, concat_bboxes, mode='iof')
                concat_ignore_overlaps = set_index(concat_ignore_overlaps, concat_bboxes_ignore_mask, -1)
                concat_ignore_max_overlaps, _ = concat_ignore_overlaps.max(dim=1)
            concat_ignore_mask = concat_ignore_max_overlaps > self.ignore_iof_thr
            concat_overlaps = set_index(concat_overlaps, concat_ignore_mask, -1)
        concat_assign_result = self.concat_assign_wrt_overlaps(concat_overlaps, concat_gt_labels, concat_num_gts)
        return concat_assign_result

    def concat_assign_wrt_overlaps(self, concat_overlaps, concat_gt_labels=None, concat_num_gts=None):
        num_images, num_gts, num_bboxes = concat_overlaps.size()
        concat_max_overlaps, concat_argmax_overlaps = concat_overlaps.max(dim=1)
        if isinstance(self.neg_iou_thr, float):
            concat_neg_mask = ((concat_max_overlaps >= 0) & (concat_max_overlaps < self.neg_iou_thr)).int()
        elif isinstance(self.neg_iou_thr, tuple):
            assert len(self.neg_iou_thr) == 2
            concat_neg_mask = ((concat_max_overlaps >= self.neg_iou_thr[0]) & (concat_max_overlaps < self.neg_iou_thr[1])).int()
        else:
            concat_neg_mask = torch.zeros(concat_max_overlaps.size(), dtype=torch.int, device=concat_max_overlaps.device)
        concat_pos_mask = (concat_max_overlaps >= self.pos_iou_thr).int()
        if self.match_low_quality:
            concat_gt_max_overlaps, concat_gt_argmax_overlaps = concat_overlaps.max(dim=2)
            concat_index_bool = (concat_gt_max_overlaps >= self.min_pos_iou) & (concat_gt_max_overlaps > 0)
            if self.gt_max_assign_all:
                pos_inds_low_quality = (concat_overlaps == concat_gt_max_overlaps.unsqueeze(2)) & concat_index_bool.unsqueeze(2)
                for i in range(num_gts):
                    pos_inds_low_quality_gt = pos_inds_low_quality[:, i, :]
                    concat_argmax_overlaps[pos_inds_low_quality_gt] = i
                    concat_pos_mask[pos_inds_low_quality_gt] = 1
            else:
                index_temp = torch.arange(0, num_gts, device=concat_max_overlaps.device)
                for index_image in range(num_images):
                    gt_argmax_overlaps = concat_gt_argmax_overlaps[index_image]
                    index_bool = concat_index_bool[index_image]
                    pos_inds_low_quality = gt_argmax_overlaps[index_bool]
                    concat_argmax_overlaps[index_image][pos_inds_low_quality] = index_temp[index_bool]
                    concat_pos_mask[index_image][pos_inds_low_quality] = 1
        concat_neg_mask = concat_neg_mask * (1 - concat_pos_mask)
        if concat_gt_labels is not None:
            concat_anchor_gt_labels = torch.zeros((num_images, num_bboxes),
                                                  dtype=concat_gt_labels.dtype,
                                                  device=concat_gt_labels.device)
            for index_image in range(num_images):
                concat_anchor_gt_labels[index_image] = torch.index_select(concat_gt_labels[index_image], 0,
                                                                          concat_argmax_overlaps[index_image])
        else:
            concat_anchor_gt_labels = None
        return AscendAssignResult(concat_num_gts, concat_pos_mask, concat_neg_mask,
                                  concat_max_overlaps, concat_argmax_overlaps, concat_anchor_gt_labels)
