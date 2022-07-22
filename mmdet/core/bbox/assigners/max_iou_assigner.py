# Copyright (c) OpenMMLab. All rights reserved.
import torch

from ..builder import BBOX_ASSIGNERS
from ..iou_calculators import build_iou_calculator
from .assign_result import AssignResult
from .base_assigner import BaseAssigner


@BBOX_ASSIGNERS.register_module()
class MaxIoUAssigner(BaseAssigner):
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

    def assign(self, bboxes, gt_bboxes, gt_bboxes_ignore=None, gt_labels=None):
        """将 gt_bboxes 按规则分配给 bbox.

        在 一阶段网络 或 多阶段的RPN 中,bbox一般为anchor.除此之外,一般为proposal
        本质上来说proposal与anchor的意义并没有什么分别,只是二者来源不同.
        anchor为固定的静态的,而proposal是基于前者进行微调从而是动态的
        该方法为每个bbox(proposal/anchor)分配一个gt bbox,每个bbox∈[-1,len(cls)-1].
        -1意味着负样本,其余意味着该gt_bboxex代表的类别索引(从0开始).
        分配过程按以下步骤完成,注意其中顺序.

        1. 将每个 bbox 归类为背景
        2. 将哪些与所有 gts 的iou都小于 neg_iou_thr的bbox都计为0
        3. 对于每个 bbox，如果与其最近的gt 的iou  >= pos_iou_thr，则将该gt类别分配给该 bbox
        4. for each gt bbox, assign its nearest proposals (may be more than
           one) to itself
4
        Args:
            bboxes (Tensor): Bounding boxes to be assigned, shape(n, 4).
            gt_bboxes (Tensor): Groundtruth boxes, shape (k, 4).
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO.
            gt_labels (Tensor, optional): Label of gt_bboxes, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.

        Example:
            >>> self = MaxIoUAssigner(0.5, 0.5)
            >>> bboxes = torch.Tensor([[0, 0, 10, 10], [10, 10, 20, 20]])
            >>> gt_bboxes = torch.Tensor([[0, 0, 10, 9]])
            >>> assign_result = self.assign(bboxes, gt_bboxes)
            >>> expected_gt_inds = torch.LongTensor([1, 0])
            >>> assert torch.all(assign_result.gt_inds == expected_gt_inds)
        """
        assign_on_cpu = True if (self.gpu_assign_thr > 0) and (
            gt_bboxes.shape[0] > self.gpu_assign_thr) else False
        # 当gt_box很多时,在CPU上计算IOU以及分配gt_box.1节省显存.2保持速度不变
        if assign_on_cpu:
            device = bboxes.device
            bboxes = bboxes.cpu()
            gt_bboxes = gt_bboxes.cpu()
            if gt_bboxes_ignore is not None:
                gt_bboxes_ignore = gt_bboxes_ignore.cpu()
            if gt_labels is not None:
                gt_labels = gt_labels.cpu()

        overlaps = self.iou_calculator(gt_bboxes, bboxes)

        if (self.ignore_iof_thr > 0 and gt_bboxes_ignore is not None
                and gt_bboxes_ignore.numel() > 0 and bboxes.numel() > 0):
            if self.ignore_wrt_candidates:
                ignore_overlaps = self.iou_calculator(
                    bboxes, gt_bboxes_ignore, mode='iof')
                ignore_max_overlaps, _ = ignore_overlaps.max(dim=1)
            else:
                ignore_overlaps = self.iou_calculator(
                    gt_bboxes_ignore, bboxes, mode='iof')
                ignore_max_overlaps, _ = ignore_overlaps.max(dim=0)
            overlaps[:, ignore_max_overlaps > self.ignore_iof_thr] = -1

        assign_result = self.assign_wrt_overlaps(overlaps, gt_labels)
        if assign_on_cpu:
            assign_result.gt_inds = assign_result.gt_inds.to(device)
            assign_result.max_overlaps = assign_result.max_overlaps.to(device)
            if assign_result.labels is not None:
                assign_result.labels = assign_result.labels.to(device)
        return assign_result

    def assign_wrt_overlaps(self, overlaps, gt_labels=None):
        """Assign w.r.t. the overlaps of bboxes with gts.

        Args:
            overlaps (Tensor): k个gt与n个box之间的iou,shape(k, n).
            gt_labels (Tensor, optional): k个gt的label, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.
        """
        num_gts, num_bboxes = overlaps.size(0), overlaps.size(1)

        # 步骤 1. 初始化为-1(背景)
        assigned_gt_inds = overlaps.new_full((num_bboxes, ),
                                             -1,
                                             dtype=torch.long)

        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = overlaps.new_zeros((num_bboxes, ))
            if num_gts == 0:
                # 没有gt,所有anchor都初始化为背景
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

        # gt为k个,anchor为n个.由于overlaps = iou(k,n)
        # 所以.max(dim=0)的两个返回值shape都为(n,).n个值代表
        # 单个anchor与所有gt的最大iou,最大iou的gt索引
        max_overlaps, argmax_overlaps = overlaps.max(dim=0)
        # 同上,这里的返回shape为(k,).以及k个值代表
        # 单个gt与所有anchor的最大iou,最大iou的anchor索引
        gt_max_overlaps, gt_argmax_overlaps = overlaps.max(dim=1)

        # 步骤 2. 分配负样本,并将其设置为0
        if isinstance(self.neg_iou_thr, float):
            assigned_gt_inds[(max_overlaps >= 0)
                             & (max_overlaps < self.neg_iou_thr)] = 0
        elif isinstance(self.neg_iou_thr, tuple):
            assert len(self.neg_iou_thr) == 2
            assigned_gt_inds[(max_overlaps >= self.neg_iou_thr[0])
                             & (max_overlaps < self.neg_iou_thr[1])] = 0

        # 步骤 3. 分配正样本: 高于pos_iou_thr的anchor.注意这里的正样本值属于[1,len(class)]
        pos_inds = max_overlaps >= self.pos_iou_thr
        assigned_gt_inds[pos_inds] = argmax_overlaps[pos_inds] + 1

        if self.match_low_quality:
            # 此处将覆盖在步骤 3 中分配的一些正样本.因此,在这里分配的 gt 对于anchor来说可能不是最合适的gt
            # 例如,如果 bbox A 与 GT bbox 1 和 2 有 0.9 和 0.8 的iou,则 bbox 1 将在步骤 3 中被指定为 bbox A 的最佳GT
            # 但是,如果 GT bbox 2 的最佳anchor是A,则 bbox A 的最佳GT将被bbox 2覆盖
            # 这可能是它没有在 ROI Heads 中使用的原因.
            for i in range(num_gts):
                if gt_max_overlaps[i] >= self.min_pos_iou:
                    if self.gt_max_assign_all:
                        max_iou_inds = overlaps[i, :] == gt_max_overlaps[i]
                        assigned_gt_inds[max_iou_inds] = i + 1
                    else:
                        assigned_gt_inds[gt_argmax_overlaps[i]] = i + 1

        if gt_labels is not None:  # 由于负样本在assigned_gt_inds上记为0,所以只能通过>0的值得gt索引来获取label
            assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), -1)
            pos_inds = torch.nonzero(
                assigned_gt_inds > 0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None

        # assigned_gt_inds代表所有box对应的样本属性[-1,len(class)],其中-1为背景,0为负样本,其余为正样本索引
        # max_overlaps代表单个anchor与所有gt的最大iou,shape为(n,)
        # assigned_labels 代表所有box对应的样本属性[-1,len(class)-1],其中-1为背景或负样本,其余为正样本所属类别
        return AssignResult(
            num_gts, assigned_gt_inds, max_overlaps, labels=assigned_labels)
