# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch
from mmengine.structures import InstanceData

from mmdet.registry import TASK_UTILS
from mmdet.structures.bbox import bbox_xyxy_to_cxcywh
from mmdet.utils import ConfigType
from .assign_result import AssignResult
from .base_assigner import BaseAssigner


@TASK_UTILS.register_module()
class UniformAssigner(BaseAssigner):
    """在prior和gt之间的Uniform Matching, 可以在正prior中达到平衡,
    并且暂时不考虑 gt_bboxes_ignore.

    Args:
        pos_ignore_thr (float): iou高于该值的负样本将被强制认为是忽略样本
        neg_ignore_thr (float): iou低于该值的正样本将被强制认为是忽略样本
        match_times(int): 离gt最近的若干个anchor/pred_box会被视为正样本.默认为4.
        iou_calculator (:obj:`ConfigDict` or dict): 计算iou的配置
    """

    def __init__(self,
                 pos_ignore_thr: float,
                 neg_ignore_thr: float,
                 match_times: int = 4,
                 iou_calculator: ConfigType = dict(type='BboxOverlaps2D')):
        self.match_times = match_times
        self.pos_ignore_thr = pos_ignore_thr
        self.neg_ignore_thr = neg_ignore_thr
        self.iou_calculator = TASK_UTILS.build(iou_calculator)

    def assign(
            self,
            pred_instances: InstanceData,
            gt_instances: InstanceData,
            gt_instances_ignore: Optional[InstanceData] = None
    ) -> AssignResult:
        """Assign gt to priors.

        The assignment is done in following steps

        1. assign -1 by default
        2. compute the L1 cost between boxes. Note that we use priors and
           predict boxes both
        3. compute the ignore indexes use gt_bboxes and predict boxes
        4. compute the ignore indexes of positive sample use priors and
           predict boxes


        Args:
            pred_instances (:obj:`InstaceData`): Instances of model
                predictions. It includes ``priors``, and the priors can
                be priors, points, or bboxes predicted by the model,
                shape(n, 4).
            gt_instances (:obj:`InstaceData`): Ground truth of instance
                annotations. It usually includes ``bboxes`` and ``labels``
                attributes.
            gt_instances_ignore (:obj:`InstaceData`, optional): Instances
                to be ignored during training. It includes ``bboxes``
                attribute data that is ignored during training and testing.
                Defaults to None.

        Returns:
            :obj:`AssignResult`: The assign result.
        """

        gt_bboxes = gt_instances.bboxes
        gt_labels = gt_instances.labels
        priors = pred_instances.priors
        bbox_pred = pred_instances.decoder_priors

        num_gts, num_bboxes = gt_bboxes.size(0), bbox_pred.size(0)

        # 1. assigned_gt_inds代表gt index(1-base), 0为负样本,-1为忽略样本
        #    assigned_labels代表gt label(0-base), 默认-1为背景
        assigned_gt_inds = bbox_pred.new_full((num_bboxes, ),
                                              0,
                                              dtype=torch.long)
        assigned_labels = bbox_pred.new_full((num_bboxes, ),
                                             -1,
                                             dtype=torch.long)
        if num_gts == 0 or num_bboxes == 0:
            # 无gt或预测框, 返回空的分配结果
            if num_gts == 0:
                # 无gt, 预测框归为背景类.但是不理解这里为什么要这么做.因为初始化时已经为0了
                assigned_gt_inds[:] = 0
            assign_result = AssignResult(
                num_gts, assigned_gt_inds, None, labels=assigned_labels)
            assign_result.set_extra_property(
                'pos_idx', bbox_pred.new_empty(0, dtype=torch.bool))
            assign_result.set_extra_property('pos_predicted_boxes',
                                             bbox_pred.new_empty((0, 4)))
            assign_result.set_extra_property('target_boxes',
                                             bbox_pred.new_empty((0, 4)))
            return assign_result

        # 2. 计算box与gt之间、prior与gt之间的L1距离, [num_box, num_gt] [num_prior, num_gt]
        cost_bbox = torch.cdist(
            bbox_xyxy_to_cxcywh(bbox_pred),
            bbox_xyxy_to_cxcywh(gt_bboxes),
            p=1)
        cost_bbox_priors = torch.cdist(
            bbox_xyxy_to_cxcywh(priors), bbox_xyxy_to_cxcywh(gt_bboxes), p=1)

        # 我们发现topk函数在cpu和cuda模式下结果不同.为了保证与源码的一致性,我们也使用cpu模式.
        # TODO: Check whether the performance of cpu and cuda are the same.
        C = cost_bbox.cpu()
        C1 = cost_bbox_priors.cpu()

        # [self.match_times, num_gt]
        index = torch.topk(
            C,  # c=b,n,x c[i]=n,x
            k=self.match_times,
            dim=0,
            largest=False)[1]  # top_k返回(val,ind),因为要取索引所以取[1]

        # [self.match_times, num_gt]
        index1 = torch.topk(C1, k=self.match_times, dim=0, largest=False)[1]
        # [self.match_times * 2*num_gt]
        indexes = torch.cat((index, index1),
                            dim=1).reshape(-1).to(bbox_pred.device)

        pred_overlaps = self.iou_calculator(bbox_pred, gt_bboxes)
        anchor_overlaps = self.iou_calculator(priors, gt_bboxes)
        # 单个pred与所有gt的最大iou, [num_box, ]
        pred_max_overlaps, _ = pred_overlaps.max(dim=1)
        # 单个gt与所有anchor的最大iou, [num_gt, ]
        anchor_max_overlaps, _ = anchor_overlaps.max(dim=0)

        # 3.在所有负样本中,将 iou 高于 0.75 的负样本(box)强制认为是忽略样本
        # 可能该gt比较大,导致一些回归出来的box都能够和该 gt有较高的IOU,那么这些box就不适合作为负样本了
        # 注意这里是先对满足忽略样本条件的样本赋予-1,是为了后续正样本(box)可以覆盖它.
        ignore_idx = pred_max_overlaps > self.neg_ignore_thr
        assigned_gt_inds[ignore_idx] = -1

        # 4.在所有正样本(prior+box)中,将 iou 低于 0.15 的正样本强制认为是忽略样本
        # 由于存在极端比例物体和小物体,在所有正样本(anchor+box)中,将IOU低于 0.15 的正样本
        # (因为不管匹配情况,top_k 都会选择出指定数目的正样本)强制认为是忽略样本
        pos_gt_index = torch.arange(
            0, C1.size(1),
            device=bbox_pred.device).repeat(self.match_times * 2)
        pos_ious = anchor_overlaps[indexes, pos_gt_index]
        pos_ignore_idx = pos_ious < self.pos_ignore_thr

        pos_gt_index_with_ignore = pos_gt_index + 1
        pos_gt_index_with_ignore[pos_ignore_idx] = -1
        assigned_gt_inds[indexes] = pos_gt_index_with_ignore

        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), -1)
            pos_inds = torch.nonzero(
                assigned_gt_inds > 0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None

        assign_result = AssignResult(
            num_gts,
            assigned_gt_inds,
            anchor_max_overlaps,  # 注意 该值非单个anchor/box的最大IOU而是gt的
            labels=assigned_labels)
        # pos_idx是一个[match_times*2*num_gt,]形状的mask,
        # 因为需要过滤掉iou小于pos_ignore_thr的部分,需要对pos_ignore_idx取反操作~
        # 同时它可以理解为是pos_predicted_boxes与target_boxes的有效索引,
        # 因为这两者可能包含了IOU低于0.15的忽略样本
        assign_result.set_extra_property('pos_idx', ~pos_ignore_idx)
        assign_result.set_extra_property('pos_predicted_boxes',
                                         bbox_pred[indexes])
        assign_result.set_extra_property('target_boxes',
                                         gt_bboxes[pos_gt_index])
        return assign_result
