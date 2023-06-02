# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Union

import torch
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData

from mmdet.registry import TASK_UTILS
from .assign_result import AssignResult
from .max_iou_assigner import MaxIoUAssigner


@TASK_UTILS.register_module()
class ApproxMaxIoUAssigner(MaxIoUAssigner):
    """Assign a corresponding gt bbox or background to each bbox.

    Each proposals will be assigned with an integer indicating the ground truth
     index. (semi-positive index: gt label (0-based), -1: background)

    - -1: negative sample, no assigned gt
    - semi-positive integer: positive sample, index (0-based) of assigned gt

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
        match_low_quality (bool): Whether to allow quality matches. This is
            usually allowed for RPN and single stage detectors, but not allowed
            in the second stage.
        gpu_assign_thr (int): The upper bound of the number of GT for GPU
            assign. When the number of gt is above this threshold, will assign
            on CPU device. Negative values mean not assign on CPU.
        iou_calculator (:obj:`ConfigDict` or dict): Config of overlaps
            Calculator.
    """

    def __init__(
        self,
        pos_iou_thr: float,
        neg_iou_thr: Union[float, tuple],
        min_pos_iou: float = .0,
        gt_max_assign_all: bool = True,
        ignore_iof_thr: float = -1,
        ignore_wrt_candidates: bool = True,
        match_low_quality: bool = True,
        gpu_assign_thr: int = -1,
        iou_calculator: Union[ConfigDict, dict] = dict(type='BboxOverlaps2D')
    ) -> None:
        self.pos_iou_thr = pos_iou_thr
        self.neg_iou_thr = neg_iou_thr
        self.min_pos_iou = min_pos_iou
        self.gt_max_assign_all = gt_max_assign_all
        self.ignore_iof_thr = ignore_iof_thr
        self.ignore_wrt_candidates = ignore_wrt_candidates
        self.gpu_assign_thr = gpu_assign_thr
        self.match_low_quality = match_low_quality
        self.iou_calculator = TASK_UTILS.build(iou_calculator)

    def assign(self,
               pred_instances: InstanceData,
               gt_instances: InstanceData,
               gt_instances_ignore: Optional[InstanceData] = None,
               **kwargs) -> AssignResult:
        """Assign gt to approxs.

        This method assign a gt bbox to each group of approxs (bboxes),
        each group of approxs is represent by a base approx (bbox) and
        will be assigned with -1, or a semi-positive number.
        background_label (-1) means negative sample,
        semi-positive number is the index (0-based) of assigned gt.
        The assignment is done in following steps, the order matters.

        1. assign every bbox to background_label (-1)
        2. use the max IoU of each group of approxs to assign
        2. assign proposals whose iou with all gts < neg_iou_thr to background
        3. for each bbox, if the iou with its nearest gt >= pos_iou_thr,
           assign it to that bbox
        4. for each gt bbox, assign its nearest proposals (may be more than
           one) to itself

        Args:
            pred_instances (:obj:`InstanceData`): Instances of model
                predictions. It includes ``priors``, and the priors can
                be anchors or points, or the bboxes predicted by the
                previous stage, has shape (n, 4). ``approxs`` means the
                group of approxs aligned with ``priors``, has shape
                (n, num_approxs, 4).
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It usually includes ``bboxes``, with shape (k, 4),
                and ``labels``, with shape (k, ).
            gt_instances_ignore (:obj:`InstanceData`, optional): Instances
                to be ignored during training. It includes ``bboxes``
                attribute data that is ignored during training and testing.
                Defaults to None.

        Returns:
            :obj:`AssignResult`: The assign result.
        """
        squares = pred_instances.priors
        approxs = pred_instances.approxs
        gt_bboxes = gt_instances.bboxes
        gt_labels = gt_instances.labels
        gt_bboxes_ignore = None if gt_instances_ignore is None else \
            gt_instances_ignore.get('bboxes', None)
        approxs_per_octave = approxs.size(1)

        num_squares = squares.size(0)
        num_gts = gt_bboxes.size(0)

        if num_squares == 0 or num_gts == 0:
            # No predictions and/or truth, return empty assignment
            overlaps = approxs.new(num_gts, num_squares)
            assign_result = self.assign_wrt_overlaps(overlaps, gt_labels)
            return assign_result

        # re-organize anchors by approxs_per_octave x num_squares
        approxs = torch.transpose(approxs, 0, 1).contiguous().view(-1, 4)
        assign_on_cpu = True if (self.gpu_assign_thr > 0) and (
            num_gts > self.gpu_assign_thr) else False
        # compute overlap and assign gt on CPU when number of GT is large
        if assign_on_cpu:
            device = approxs.device
            approxs = approxs.cpu()
            gt_bboxes = gt_bboxes.cpu()
            if gt_bboxes_ignore is not None:
                gt_bboxes_ignore = gt_bboxes_ignore.cpu()
            if gt_labels is not None:
                gt_labels = gt_labels.cpu()
        all_overlaps = self.iou_calculator(approxs, gt_bboxes)

        overlaps, _ = all_overlaps.view(approxs_per_octave, num_squares,
                                        num_gts).max(dim=0)
        overlaps = torch.transpose(overlaps, 0, 1)

        if (self.ignore_iof_thr > 0 and gt_bboxes_ignore is not None
                and gt_bboxes_ignore.numel() > 0 and squares.numel() > 0):
            if self.ignore_wrt_candidates:
                ignore_overlaps = self.iou_calculator(
                    squares, gt_bboxes_ignore, mode='iof')
                ignore_max_overlaps, _ = ignore_overlaps.max(dim=1)
            else:
                ignore_overlaps = self.iou_calculator(
                    gt_bboxes_ignore, squares, mode='iof')
                ignore_max_overlaps, _ = ignore_overlaps.max(dim=0)
            overlaps[:, ignore_max_overlaps > self.ignore_iof_thr] = -1

        assign_result = self.assign_wrt_overlaps(overlaps, gt_labels)
        if assign_on_cpu:
            assign_result.gt_inds = assign_result.gt_inds.to(device)
            assign_result.max_overlaps = assign_result.max_overlaps.to(device)
            if assign_result.labels is not None:
                assign_result.labels = assign_result.labels.to(device)
        return assign_result
