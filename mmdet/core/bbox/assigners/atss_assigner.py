# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch

from ..builder import BBOX_ASSIGNERS
from ..iou_calculators import build_iou_calculator
from .assign_result import AssignResult
from .base_assigner import BaseAssigner


@BBOX_ASSIGNERS.register_module()
class ATSSAssigner(BaseAssigner):
    """为每个box分配gt或者bg,同时分配一个gt label.

    - 0: 负样本, 未分配 gt
    - 正整数: 它代表了gt box的索引(1-base)

    如果 `alpha` 不为 None, 表示采用dynamic cost ATSSAssigner,目前仅在 DDOD 中使用.

    Args:
        topk (float): 各层级上选择的box数量
    """

    def __init__(self,
                 topk,
                 alpha=None,
                 iou_calculator=dict(type='BboxOverlaps2D'),
                 ignore_iof_thr=-1):
        self.topk = topk
        self.alpha = alpha
        self.iou_calculator = build_iou_calculator(iou_calculator)
        self.ignore_iof_thr = ignore_iof_thr

    """为每个box分配gt或者bg.

    Args:
        topk (int): 各层级上选择的box数量.
        alpha (float): 每个proposal的cost rate参数,仅在DDOD中使用.
        iou_calculator (dict): IOU计算方式的配置字典.
        ignore_iof_thr (int): 是否忽略最大重叠overlaps.Default -1 (1 or -1).
    """

    # https://github.com/sfzhang15/ATSS/blob/master/atss_core/modeling/rpn/atss/loss.py
    def assign(self,
               bboxes,
               num_level_bboxes,
               gt_bboxes,
               gt_bboxes_ignore=None,
               gt_labels=None,
               cls_scores=None,
               bbox_preds=None):
        """将 gt 分配给 box.

        分配按以下步骤完成

        1. 计算box与gt的iou
        2. 计算box与gt之间的中心距离
        3. 在每个层级上,对于每个 gt,选择中心最接近 gt 中心的 k 个box,
           因此我们总共选择 k*num_level个box 作为每个 gt 的候选box
        4. 得到这些候选box对应的iou,计算mean和std,令mean+std为iou阈值
        5. 选择那些iou大于等于iou阈值的候选box为正样本
        6. 将正样本的中心限制在gt内部

        如果 alpha、cls_scores、bbox_preds 都不为 None,
        第一步中的overlaps计算还将包括dynamic cost,不过目前仅在 DDOD 中使用.

        Args:
            bboxes (Tensor): 要被分配gt的box, (n, 4).
            num_level_bboxes (List): 各层级上合格的box数量,其总和等于n
            gt_bboxes (Tensor): gt box, (k, 4).
            gt_bboxes_ignore (Tensor, optional): 被忽略的gt box,比如在COCO里的crowd属性.
            gt_labels (Tensor, optional): gt label, (k, ).
            cls_scores (list[Tensor]): 所有层级的cls_score,
                [bs, na * num_cls,h,w] * num_level.默认 None.
            bbox_preds (list[Tensor]): 所有层级的box_reg,
                [bs, na * 4,h,w] * num_level.默认 None.

        Returns:
            :obj:`AssignResult`: 分配结果.
        """
        INF = 100000000
        bboxes = bboxes[:, :4]
        num_gt, num_bboxes = gt_bboxes.size(0), bboxes.size(0)

        message = '参数错误.如果你想使用cost-based 的ATSSAssigner,请同时设置cls_scores, ' \
                  'bbox_preds 以及 self.alpha. '

        if self.alpha is None:
            # 普通的ATSSAssigner
            overlaps = self.iou_calculator(bboxes, gt_bboxes)
            if cls_scores is not None or bbox_preds is not None:
                warnings.warn(message)
        else:
            # DDOD中的Dynamic cost ATSSAssigner
            assert cls_scores is not None and bbox_preds is not None, message

            # compute cls cost for bbox and GT
            cls_cost = torch.sigmoid(cls_scores[:, gt_labels])

            # compute iou between all bbox and gt
            overlaps = self.iou_calculator(bbox_preds, gt_bboxes)

            # make sure that we are in element-wise multiplication
            assert cls_cost.shape == overlaps.shape

            # overlaps is actually a cost matrix
            overlaps = cls_cost**(1 - self.alpha) * overlaps**self.alpha

        # 初始化为0, 默认全都是负样本.正整数为gt box索引(1-base)
        assigned_gt_inds = overlaps.new_full((num_bboxes, ),
                                             0,
                                             dtype=torch.long)

        if num_gt == 0 or num_bboxes == 0:
            # 无gt或者box, 返回一个空的分配结果
            max_overlaps = overlaps.new_zeros((num_bboxes, ))
            if num_gt == 0:
                # 没有gt,那么一切都设置为负样本
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = overlaps.new_full((num_bboxes, ),
                                                    -1,
                                                    dtype=torch.long)
            return AssignResult(
                num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)

        # 计算box和gt之间的中心距离
        gt_cx = (gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2.0
        gt_cy = (gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2.0
        gt_points = torch.stack((gt_cx, gt_cy), dim=1)

        bboxes_cx = (bboxes[:, 0] + bboxes[:, 2]) / 2.0
        bboxes_cy = (bboxes[:, 1] + bboxes[:, 3]) / 2.0
        bboxes_points = torch.stack((bboxes_cx, bboxes_cy), dim=1)

        distances = (bboxes_points[:, None, :] -
                     gt_points[None, :, :]).pow(2).sum(-1).sqrt()

        # 如果存在gt_bboxes_ignore,并且存在满足忽略条件的box.那么让这些box的所对应的
        # gt索引为-1以及与所有gt的距离为INF.此举会导致这些box在后续忽略计算Loss
        if (self.ignore_iof_thr > 0 and gt_bboxes_ignore is not None
                and gt_bboxes_ignore.numel() > 0 and bboxes.numel() > 0):
            ignore_overlaps = self.iou_calculator(
                bboxes, gt_bboxes_ignore, mode='iof')
            ignore_max_overlaps, _ = ignore_overlaps.max(dim=1)
            ignore_idxs = ignore_max_overlaps > self.ignore_iof_thr
            distances[ignore_idxs, :] = INF
            assigned_gt_inds[ignore_idxs] = -1

        # 根据中心距离选择候选box
        candidate_idxs = []
        start_idx = 0
        for level, bboxes_per_level in enumerate(num_level_bboxes):
            # 在每个层级,对于每个gt,选择中心最接近gt中心的k个box
            end_idx = start_idx + bboxes_per_level
            distances_per_level = distances[start_idx:end_idx, :]
            selectable_k = min(self.topk, bboxes_per_level)
            # 单个gt与所有box的前k个最大iou * num_gt -> [top_k, num_gt]
            _, topk_idxs_per_level = distances_per_level.topk(
                selectable_k, dim=0, largest=False)
            candidate_idxs.append(topk_idxs_per_level + start_idx)
            start_idx = end_idx
        # 注意此时的索引是基于所有层级的, [num_level*top_k, num_gt]
        candidate_idxs = torch.cat(candidate_idxs, dim=0)

        # 得到这些候选box对应的iou,计算mean和std,设置mean+std为iou阈值
        # 注意,作为索引torch.arange(num_gt)会被广播至candidate_idxs相同维度
        candidate_overlaps = overlaps[candidate_idxs, torch.arange(num_gt)]
        overlaps_mean_per_gt = candidate_overlaps.mean(0)
        overlaps_std_per_gt = candidate_overlaps.std(0)
        overlaps_thr_per_gt = overlaps_mean_per_gt + overlaps_std_per_gt

        is_pos = candidate_overlaps >= overlaps_thr_per_gt[None, :]

        # 将正样本的中心限制在 gt内部
        for gt_idx in range(num_gt):
            # 下面会将bboxes_cx等由[num_box, ]扩充至[num_gt*num_box, ]
            # 所以下面作为其索引的candidate_idx也需要更新一下索引范围由[0, num_box) -> [0, num_gt*num_box)
            candidate_idxs[:, gt_idx] += gt_idx * num_bboxes
        ep_bboxes_cx = bboxes_cx.view(1, -1).expand(
            num_gt, num_bboxes).contiguous().view(-1)
        ep_bboxes_cy = bboxes_cy.view(1, -1).expand(
            num_gt, num_bboxes).contiguous().view(-1)
        # [num_level*top_k, num_gt] -> [num_level * top_k * num_gt, ]
        candidate_idxs = candidate_idxs.view(-1)

        # 计算正样本中心到gt边界的四个方向距离
        # cx -> [num_gt * num_bboxes] idx -> [num_level * top_k * num_gt]
        # 各个方向的距离, -> [num_level * top_k, num_gt]
        l_ = ep_bboxes_cx[candidate_idxs].view(-1, num_gt) - gt_bboxes[:, 0]
        t_ = ep_bboxes_cy[candidate_idxs].view(-1, num_gt) - gt_bboxes[:, 1]
        r_ = gt_bboxes[:, 2] - ep_bboxes_cx[candidate_idxs].view(-1, num_gt)
        b_ = gt_bboxes[:, 3] - ep_bboxes_cy[candidate_idxs].view(-1, num_gt)
        is_in_gts = torch.stack([l_, t_, r_, b_], dim=1).min(dim=1)[0] > 0.01

        # 同时满足iou阈值以及在gt内部的mask [num_level*top_k, num_gt]
        is_pos = is_pos & is_in_gts
        # 1.先将iou初始化为-INF 2.转置再展平是为了使下面的is_pos能够顺利获取到iou
        # [num_box, num_gt] -> [num_gt, num_box] -> [num_gt*num_box, ]
        overlaps_inf = torch.full_like(overlaps,
                                       -INF).t().contiguous().view(-1)
        # 满足上述两个条件的box索引,记其shape为[num_pos, ], ∈[0, num_gt*num_box)
        index = candidate_idxs[is_pos.view(-1)]
        # .t().view(-1) -> [num_gt * num_box, ],因为index的取值范围,所以该操作是必要的
        overlaps_inf[index] = overlaps.t().contiguous().view(-1)[index]
        # 转置回[num_box, num_gt]
        overlaps_inf = overlaps_inf.view(num_gt, -1).t()

        # 如果一个box被分配给了多个gt,那么就选择与该box最高iou的那个gt.
        max_overlaps, argmax_overlaps = overlaps_inf.max(dim=1)
        assigned_gt_inds[
            max_overlaps != -INF] = argmax_overlaps[max_overlaps != -INF] + 1

        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), -1)
            pos_inds = torch.nonzero(
                assigned_gt_inds > 0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                # 由于assigned_gt_inds中正样本是1-base,所以需要-1
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None
        # assigned_gt_inds: 所有box对应的gt index, 0 -> 负样本. >= 1 正样本索引(1-base)
        # max_overlaps: 所有box对应最大IOU值, 代表该box与所有gt的最大IOU值
        # assigned_labels:  所有box对应的gt label, -1 -> 负样本. >= 0 正样本label
        return AssignResult(
            num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)
