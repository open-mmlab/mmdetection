# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple, Union

import torch
from mmengine.structures import InstanceData

from mmdet.registry import TASK_UTILS
from mmdet.utils import ConfigType
from .assign_result import AssignResult
from .base_assigner import BaseAssigner


@TASK_UTILS.register_module()
class GridAssigner(BaseAssigner):
    """Assign a corresponding gt bbox or background to each bbox.

    Each proposals will be assigned with `-1`, `0`, or a positive integer
    indicating the ground truth index.

    - -1: don't care
    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:
        pos_iou_thr (float): IoU threshold for positive bboxes.
        neg_iou_thr (float or tuple[float, float]): IoU threshold for negative
        bboxes.
        min_pos_iou (float): Minimum iou for a bbox to be considered as a
            positive bbox. Positive samples can have smaller IoU than
            pos_iou_thr due to the 4th step (assign max IoU sample to each gt).
            Defaults to 0.
        gt_max_assign_all (bool): Whether to assign all bboxes with the same
            highest overlap with some gt to that gt.
        iou_calculator (:obj:`ConfigDict` or dict): Config of overlaps
            Calculator.
    """

    def __init__(
        self,
        pos_iou_thr: float,
        neg_iou_thr: Union[float, Tuple[float, float]],
        min_pos_iou: float = .0,
        gt_max_assign_all: bool = True,
        iou_calculator: ConfigType = dict(type='BboxOverlaps2D')
    ) -> None:
        self.pos_iou_thr = pos_iou_thr
        self.neg_iou_thr = neg_iou_thr
        self.min_pos_iou = min_pos_iou
        self.gt_max_assign_all = gt_max_assign_all
        self.iou_calculator = TASK_UTILS.build(iou_calculator)

    def assign(self,
               pred_instances: InstanceData,
               gt_instances: InstanceData,
               gt_instances_ignore: Optional[InstanceData] = None,
               **kwargs) -> AssignResult:
        """Assign gt to bboxes. The process is very much like the max iou
        assigner, except that positive samples are constrained within the cell
        that the gt boxes fell in.

        This method assign a gt bbox to every bbox (proposal/anchor), each bbox
        will be assigned with -1, 0, or a positive number. -1 means don't care,
        0 means negative sample, positive number is the index (1-based) of
        assigned gt.
        The assignment is done in following steps, the order matters.

        1. assign every bbox to -1
        2. assign proposals whose iou with all gts <= neg_iou_thr to 0
        3. for each bbox within a cell, if the iou with its nearest gt >
            pos_iou_thr and the center of that gt falls inside the cell,
            assign it to that bbox
        4. for each gt bbox, assign its nearest proposals within the cell the
            gt bbox falls in to itself.

        Args:
            pred_instances (:obj:`InstanceData`): Instances of model
                predictions. It includes ``priors``, and the priors can
                be anchors or points, or the bboxes predicted by the
                previous stage, has shape (n, 4). The bboxes predicted by
                the current model or stage will be named ``bboxes``,
                ``labels``, and ``scores``, the same as the ``InstanceData``
                in other places.
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
        gt_bboxes = gt_instances.bboxes
        gt_labels = gt_instances.labels

        priors = pred_instances.priors
        responsible_flags = pred_instances.responsible_flags

        num_gts, num_priors = gt_bboxes.size(0), priors.size(0)

        # compute iou between all gt and priors
        overlaps = self.iou_calculator(gt_bboxes, priors)

        # 1. assign -1 by default
        assigned_gt_inds = overlaps.new_full((num_priors, ),
                                             -1,
                                             dtype=torch.long)

        if num_gts == 0 or num_priors == 0:
            # No ground truth or priors, return empty assignment
            max_overlaps = overlaps.new_zeros((num_priors, ))
            if num_gts == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            assigned_labels = overlaps.new_full((num_priors, ),
                                                -1,
                                                dtype=torch.long)
            return AssignResult(
                num_gts,
                assigned_gt_inds,
                max_overlaps,
                labels=assigned_labels)

        # 2. assign negative: below
        # for each anchor, which gt best overlaps with it
        # for each anchor, the max iou of all gts
        # shape of max_overlaps == argmax_overlaps == num_priors
        max_overlaps, argmax_overlaps = overlaps.max(dim=0)

        if isinstance(self.neg_iou_thr, float):
            assigned_gt_inds[(max_overlaps >= 0)
                             & (max_overlaps <= self.neg_iou_thr)] = 0
        elif isinstance(self.neg_iou_thr, (tuple, list)):
            assert len(self.neg_iou_thr) == 2
            assigned_gt_inds[(max_overlaps > self.neg_iou_thr[0])
                             & (max_overlaps <= self.neg_iou_thr[1])] = 0

        # 3. assign positive: falls into responsible cell and above
        # positive IOU threshold, the order matters.
        # the prior condition of comparison is to filter out all
        # unrelated anchors, i.e. not responsible_flags
        overlaps[:, ~responsible_flags.type(torch.bool)] = -1.

        # calculate max_overlaps again, but this time we only consider IOUs
        # for anchors responsible for prediction
        max_overlaps, argmax_overlaps = overlaps.max(dim=0)

        # for each gt, which anchor best overlaps with it
        # for each gt, the max iou of all proposals
        # shape of gt_max_overlaps == gt_argmax_overlaps == num_gts
        gt_max_overlaps, gt_argmax_overlaps = overlaps.max(dim=1)

        pos_inds = (max_overlaps > self.pos_iou_thr) & responsible_flags.type(
            torch.bool)
        assigned_gt_inds[pos_inds] = argmax_overlaps[pos_inds] + 1

        # 4. assign positive to max overlapped anchors within responsible cell
        for i in range(num_gts):
            if gt_max_overlaps[i] > self.min_pos_iou:
                if self.gt_max_assign_all:
                    max_iou_inds = (overlaps[i, :] == gt_max_overlaps[i]) & \
                         responsible_flags.type(torch.bool)
                    assigned_gt_inds[max_iou_inds] = i + 1
                elif responsible_flags[gt_argmax_overlaps[i]]:
                    assigned_gt_inds[gt_argmax_overlaps[i]] = i + 1

        # assign labels of positive anchors
        assigned_labels = assigned_gt_inds.new_full((num_priors, ), -1)
        pos_inds = torch.nonzero(
            assigned_gt_inds > 0, as_tuple=False).squeeze()
        if pos_inds.numel() > 0:
            assigned_labels[pos_inds] = gt_labels[assigned_gt_inds[pos_inds] -
                                                  1]

        return AssignResult(
            num_gts, assigned_gt_inds, max_overlaps, labels=assigned_labels)
