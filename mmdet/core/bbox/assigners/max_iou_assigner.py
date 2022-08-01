# Copyright (c) OpenMMLab. All rights reserved.
import torch

from ..builder import BBOX_ASSIGNERS
from ..iou_calculators import build_iou_calculator
from .assign_result import AssignResult
from .base_assigner import BaseAssigner


@BBOX_ASSIGNERS.register_module()
class MaxIoUAssigner(BaseAssigner):
    """为每个 box 分配一个对应的 gt 或 背景.

    每个box都会被分配一个label,∈[-1,len(gt)-1].

    - -1: 没有分配 gt 的负样本
    - [0,len(gt)-1]]: 代表gt索引的正样本

    Args:
        pos_iou_thr (float): 正样本 的 IoU 阈值.
        neg_iou_thr (float or tuple): 负样本 的 IoU 阈值.
        min_pos_iou (float): 将 box 视为正样本 的最小 IoU 阈值.
            由于在第 4 步(将最大 IoU 样本分配给每个 gt),正样本的 IoU 可能小于 pos_iou_thr.
            `min_pos_iou` 的设置主要为避免将与 GT 具有极小 iou 的 box 被分配为正样本.
            它在 1x(12epoch) 的训练中带来了 0.3 mAP 的提升,但不影响 3x 的性能. 更多比较参考:
            `PR #7464 <https://github.com/open-mmlab/mmdetection/pull/7464>`_.
        gt_max_assign_all (bool): 是否将 与某个gt具有相同最高IOU的所有 box 分配给该 gt.
        ignore_iof_thr (float): 忽略 box 的 IoF 阈值(如果指定了 `gt_bboxes_ignore`).
            负值意味着不忽略任何 box.
        ignore_wrt_candidates (bool): 计算iof时,True表示f为box.False表示f为gt.
        match_low_quality (bool): 是否允许低质量匹配产生. 这对于RPN和单阶段检测模型通常是允许的,
            但在二阶段是不允许的. 详情参阅步骤 4.
        gpu_assign_thr (int): GPU分配的GT数量上限. 当 gt 的数量高于此阈值时, 将在 CPU 设备上分配.
            负值表示不在 CPU 上分配.
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
        4. 对于每个 gt, 分配与其有最大IOU的box(可能不止一个)给它

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

        overlaps = self.iou_calculator(gt_bboxes, bboxes)  # 计算gt与box的IOU

        # 和gt_bboxes_ignore的IOU大于ignore_iof_thr的 所有box与所有gt的IOU都被设置-1
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
        # 迁移回GPU设备
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

        # 步骤 1. 初始化样本值为-1(背景).注:0代表负样本,正数代表分配给该box的gt索引
        assigned_gt_inds = overlaps.new_full((num_bboxes, ),
                                             -1,
                                             dtype=torch.long)

        if num_gts == 0 or num_bboxes == 0:
            # gt 或 box数量为0,则返回一个空的AssignResult
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

        # 步骤 3. 分配正样本: 高于pos_iou_thr的anchor.注意这里的正样本值属于[1,len(gt)]
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

        # assigned_gt_inds代表所有box对应的样本属性[-1,len(gt)],其中-1为背景,0为负样本,其余为分配的gt索引
        # max_overlaps代表单个anchor与所有gt的最大iou,shape为(n,)
        # assigned_labels 代表所有box对应的样本属性[-1,len(class)-1],其中-1为背景或负样本,其余为分配的gt类别
        return AssignResult(
            num_gts, assigned_gt_inds, max_overlaps, labels=assigned_labels)
