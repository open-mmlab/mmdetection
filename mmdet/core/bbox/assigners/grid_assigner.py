# Copyright (c) OpenMMLab. All rights reserved.
import torch

from ..builder import BBOX_ASSIGNERS
from ..iou_calculators import build_iou_calculator
from .assign_result import AssignResult
from .base_assigner import BaseAssigner


@BBOX_ASSIGNERS.register_module()
class GridAssigner(BaseAssigner):
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
    """

    def __init__(self,
                 pos_iou_thr,
                 neg_iou_thr,
                 min_pos_iou=.0,
                 gt_max_assign_all=True,
                 iou_calculator=dict(type='BboxOverlaps2D')):
        self.pos_iou_thr = pos_iou_thr
        self.neg_iou_thr = neg_iou_thr
        if isinstance(neg_iou_thr, float):
            self.neg_iou_thr = (-1e-6, neg_iou_thr)
        self.min_pos_iou = min_pos_iou
        self.gt_max_assign_all = gt_max_assign_all
        self.iou_calculator = build_iou_calculator(iou_calculator)

    def assign_statically(self,
                          bboxes,
                          box_responsible_flags,
                          gt_bboxes,
                          gt_labels=None):
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
            bboxes (Tensor): Bounding boxes to be assigned, shape(n, 4).
            box_responsible_flags (Tensor): flag to indicate whether box is
                responsible for prediction, shape(n, )
            gt_bboxes (Tensor): Groundtruth boxes, shape (k, 4).
            gt_labels (Tensor, optional): Label of gt_bboxes, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.
        """
        num_bboxes = bboxes.size(0)
        valid_gt_flags = gt_bboxes.abs().sum(-1) > 0
        num_gts = valid_gt_flags.sum()

        # compute iou between all gt and bboxes
        overlaps = self.iou_calculator(gt_bboxes, bboxes)
        overlaps = torch.clip(overlaps, 0.0, 1.0) # Overlap less than 0 or greater than 1 makes no sense physically
        max_overlaps, argmax_overlaps = overlaps.max(dim=0)

        # 1. assign -1 by default
        assigned_gt_inds = torch.ones((num_bboxes,), dtype=torch.long)*-1
        # assigned_gt_inds = overlaps.new_full((num_bboxes, ),
        #                                      -1,
        #                                      dtype=torch.long)

        # 2. assign negative: below
        # for each anchor, which gt best overlaps with it
        # for each anchor, the max iou of all gts
        # shape of max_overlaps == argmax_overlaps == num_bboxes
        overlapped_flags = 1-((max_overlaps>self.neg_iou_thr[0])&(max_overlaps<=self.neg_iou_thr[1])).float()
        assigned_gt_inds = assigned_gt_inds*overlapped_flags

        # 3. assign positive: falls into responsible cell and above
        # positive IOU threshold, the order matters.
        # the prior condition of comparison is to filter out all
        # unrelated anchors, i.e. not box_responsible_flags
        overlaps = overlaps * box_responsible_flags + (box_responsible_flags-1.0)

        # calculate max_overlaps again, but this time we only consider IOUs
        # for anchors responsible for prediction
        max_overlaps, argmax_overlaps = overlaps.max(dim=0)

        # for each gt, which anchor best overlaps with it
        # for each gt, the max iou of all proposals
        # shape of gt_max_overlaps == gt_argmax_overlaps == num_gts
        gt_max_overlaps, gt_argmax_overlaps = overlaps.max(dim=1)
        pos_inds = (max_overlaps >
                    self.pos_iou_thr) & box_responsible_flags.type(torch.bool)
        pos_inds = pos_inds.float()
        assigned_gt_inds = assigned_gt_inds*(1-pos_inds)+(argmax_overlaps + pos_inds)*pos_inds

        # 4. assign positive to max overlapped anchors within responsible cell
        if self.gt_max_assign_all:
            keep_by_min_pos_iou = (gt_max_overlaps > self.min_pos_iou).unsqueeze(-1)
            # keep all the predicted boxes with max IOU
            max_iou_inds = (overlaps == gt_max_overlaps.unsqueeze(-1)) & box_responsible_flags.type(torch.bool)
            max_iou_inds = keep_by_min_pos_iou & max_iou_inds
            class_num_list = torch.range(1, max_iou_inds.shape[0]).unsqueeze(-1)
            positive_inds = max_iou_inds * class_num_list
            # use the last gt box if more than one gt boxes have max IOU value
            positive_inds = torch.max(positive_inds, 0).values
        else:
            positive_flags = torch.nn.functional.one_hot(gt_argmax_overlaps,num_classes=bboxes.shape[0])
            positive_flags = positive_flags * valid_gt_flags.unsqueeze(-1) #  remove pad boxes
            class_num_list = torch.range(1, positive_flags.shape[0]).unsqueeze(-1)
            # positive_flags is generated by one_hot, so every line will have only one int(1),
            # so positive_inds will only have one idx on every line
            positive_inds = positive_flags * class_num_list
            assert (positive_inds>0).sum(1).max()==1, 'every line should have only one idx'
            positive_inds = positive_inds.max(0).values
        positive_inds_mask = (positive_inds > 0.0).float()
        assigned_gt_inds = positive_inds + assigned_gt_inds*(1-positive_inds_mask)

        # assign labels of positive anchors
        if gt_labels is not None:
            raise NotImplementedError('gt_lables assignment is not implemented statically')
        else:
            assigned_labels = None

        return AssignResult(
            num_gts, assigned_gt_inds, max_overlaps, labels=assigned_labels)

    def assign(self, bboxes, box_responsible_flags, gt_bboxes, gt_labels=None):
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
            bboxes (Tensor): Bounding boxes to be assigned, shape(n, 4).
            box_responsible_flags (Tensor): flag to indicate whether box is
                responsible for prediction, shape(n, )
            gt_bboxes (Tensor): Groundtruth boxes, shape (k, 4).
            gt_labels (Tensor, optional): Label of gt_bboxes, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.
        """
        num_gts, num_bboxes = gt_bboxes.size(0), bboxes.size(0)

        # compute iou between all gt and bboxes
        overlaps = self.iou_calculator(gt_bboxes, bboxes)

        # 1. assign -1 by default
        assigned_gt_inds = overlaps.new_full((num_bboxes, ),
                                             -1,
                                             dtype=torch.long)

        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = overlaps.new_zeros((num_bboxes, ))
            if num_gts == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = overlaps.new_full((num_bboxes, ),
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
        # shape of max_overlaps == argmax_overlaps == num_bboxes
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
        # unrelated anchors, i.e. not box_responsible_flags
        overlaps[:, ~box_responsible_flags.type(torch.bool)] = -1.

        # calculate max_overlaps again, but this time we only consider IOUs
        # for anchors responsible for prediction
        max_overlaps, argmax_overlaps = overlaps.max(dim=0)

        # for each gt, which anchor best overlaps with it
        # for each gt, the max iou of all proposals
        # shape of gt_max_overlaps == gt_argmax_overlaps == num_gts
        gt_max_overlaps, gt_argmax_overlaps = overlaps.max(dim=1)

        pos_inds = (max_overlaps >
                    self.pos_iou_thr) & box_responsible_flags.type(torch.bool)
        assigned_gt_inds[pos_inds] = argmax_overlaps[pos_inds] + 1

        # 4. assign positive to max overlapped anchors within responsible cell
        for i in range(num_gts):
            if gt_max_overlaps[i] > self.min_pos_iou:
                if self.gt_max_assign_all:
                    max_iou_inds = (overlaps[i, :] == gt_max_overlaps[i]) & \
                         box_responsible_flags.type(torch.bool)
                    assigned_gt_inds[max_iou_inds] = i + 1
                elif box_responsible_flags[gt_argmax_overlaps[i]]:
                    assigned_gt_inds[gt_argmax_overlaps[i]] = i + 1

        # assign labels of positive anchors
        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), -1)
            pos_inds = torch.nonzero(
                assigned_gt_inds > 0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1]

        else:
            assigned_labels = None

        return AssignResult(
            num_gts, assigned_gt_inds, max_overlaps, labels=assigned_labels)
