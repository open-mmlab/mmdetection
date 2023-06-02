# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Union

import torch
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.registry import TASK_UTILS
from .assign_result import AssignResult
from .base_assigner import BaseAssigner


@TASK_UTILS.register_module()
class MaxIoUAssigner(BaseAssigner):
    """为每个 box 分配一个对应的 gt 或 负样本.

    每个box都会被分配一个gt索引,∈[-1,len(gt)-1].

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
<<<<<<< HEAD:mmdet/core/bbox/assigners/max_iou_assigner.py
        gt_max_assign_all (bool): 是否将该gt索引(1-base)分配给 所有与该gt具有相同最高IOU的 box.
        ignore_iof_thr (float): 忽略 box 的 IoF 阈值(如果指定了 `gt_bboxes_ignore`).
            负值意味着不忽略任何 box.
        ignore_wrt_candidates (bool): 计算iof时,True表示f为box.False表示f为gt.
        match_low_quality (bool): 是否允许低质量匹配产生. 这对于RPN和单阶段检测模型通常是允许的,
            但在二阶段是不允许的. 详情参阅步骤 4.
        gpu_assign_thr (int): GPU分配的GT数量上限. 当 gt 的数量高于此阈值时, 将在 CPU 设备上分配.
            负值表示不在 CPU 上分配.
=======
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
        iou_calculator (dict): Config of overlaps Calculator.
>>>>>>> mmdetection/main:mmdet/models/task_modules/assigners/max_iou_assigner.py
    """

    def __init__(self,
                 pos_iou_thr: float,
                 neg_iou_thr: Union[float, tuple],
                 min_pos_iou: float = .0,
                 gt_max_assign_all: bool = True,
                 ignore_iof_thr: float = -1,
                 ignore_wrt_candidates: bool = True,
                 match_low_quality: bool = True,
                 gpu_assign_thr: float = -1,
                 iou_calculator: dict = dict(type='BboxOverlaps2D')):
        self.pos_iou_thr = pos_iou_thr
        self.neg_iou_thr = neg_iou_thr
        self.min_pos_iou = min_pos_iou
        self.gt_max_assign_all = gt_max_assign_all
        self.ignore_iof_thr = ignore_iof_thr
        self.ignore_wrt_candidates = ignore_wrt_candidates
        self.gpu_assign_thr = gpu_assign_thr
        self.match_low_quality = match_low_quality
        self.iou_calculator = TASK_UTILS.build(iou_calculator)

<<<<<<< HEAD:mmdet/core/bbox/assigners/max_iou_assigner.py
    def assign(self, bboxes, gt_bboxes, gt_bboxes_ignore=None, gt_labels=None):
        """将 gt_bboxes 按规则分配给 bbox.
=======
    def assign(self,
               pred_instances: InstanceData,
               gt_instances: InstanceData,
               gt_instances_ignore: Optional[InstanceData] = None,
               **kwargs) -> AssignResult:
        """Assign gt to bboxes.
>>>>>>> mmdetection/main:mmdet/models/task_modules/assigners/max_iou_assigner.py

        在 一阶段网络 或 多阶段的RPN 中,bbox一般为anchor或proposal
        本质上来说proposal与anchor的意义并没有什么分别,只是二者来源不同.
        anchor为固定的静态的,而proposal是基于前者进行微调从而是动态的
        该方法为每个bbox(proposal/anchor)分配一个gt bbox,每个bbox∈[-1,len(cls)-1].
        -1意味着负样本,其余意味着该gt_bboxex代表的类别索引(从0开始).
        分配过程按以下步骤完成,注意其中顺序.

        1. 将每个 bbox 归为忽略样本
        2. 将哪些与所有 gts 的iou都小于 neg_iou_thr的bbox都计为0
        3. 对于每个 bbox，如果与其最近的gt 的iou  >= pos_iou_thr，则将该gt类别分配给该 bbox
        4. 对于每个 gt, 分配与其有最大IOU的box(可能不止一个)给它

        Args:
<<<<<<< HEAD:mmdet/core/bbox/assigners/max_iou_assigner.py
            bboxes (Tensor): 要为其分配gt box 的pred box, shape(n, 4).
            gt_bboxes (Tensor): gt box, shape (k, 4).
            gt_bboxes_ignore (Tensor, optional): 被忽略的gt box,
                如果不为None,其shape为[num_ignored_gts, 4]
            gt_labels (Tensor, optional): gt_bboxes对应的label, shape (k, ).
=======
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
>>>>>>> mmdetection/main:mmdet/models/task_modules/assigners/max_iou_assigner.py

        Returns:
            :obj:`AssignResult`: 分配结果.

        Example:
            >>> from mmengine.structures import InstanceData
            >>> self = MaxIoUAssigner(0.5, 0.5)
            >>> pred_instances = InstanceData()
            >>> pred_instances.priors = torch.Tensor([[0, 0, 10, 10],
            ...                                      [10, 10, 20, 20]])
            >>> gt_instances = InstanceData()
            >>> gt_instances.bboxes = torch.Tensor([[0, 0, 10, 9]])
            >>> gt_instances.labels = torch.Tensor([0])
            >>> assign_result = self.assign(pred_instances, gt_instances)
            >>> expected_gt_inds = torch.LongTensor([1, 0])
            >>> assert torch.all(assign_result.gt_inds == expected_gt_inds)
        """
        gt_bboxes = gt_instances.bboxes
        priors = pred_instances.priors
        gt_labels = gt_instances.labels
        if gt_instances_ignore is not None:
            gt_bboxes_ignore = gt_instances_ignore.bboxes
        else:
            gt_bboxes_ignore = None

        assign_on_cpu = True if (self.gpu_assign_thr > 0) and (
            gt_bboxes.shape[0] > self.gpu_assign_thr) else False
        # 当gt_box很多时,在CPU上计算IOU以及分配gt_box.1节省显存.2保持速度不变
        if assign_on_cpu:
            device = priors.device
            priors = priors.cpu()
            gt_bboxes = gt_bboxes.cpu()
            gt_labels = gt_labels.cpu()
            if gt_bboxes_ignore is not None:
                gt_bboxes_ignore = gt_bboxes_ignore.cpu()

<<<<<<< HEAD:mmdet/core/bbox/assigners/max_iou_assigner.py
        overlaps = self.iou_calculator(gt_bboxes, bboxes)  # 计算gt与box的IOU
=======
        overlaps = self.iou_calculator(gt_bboxes, priors)
>>>>>>> mmdetection/main:mmdet/models/task_modules/assigners/max_iou_assigner.py

        # 和gt_bboxes_ignore的IOU大于ignore_iof_thr的 所有box与所有gt的IOU都被设置-1
        if (self.ignore_iof_thr > 0 and gt_bboxes_ignore is not None
                and gt_bboxes_ignore.numel() > 0 and priors.numel() > 0):
            if self.ignore_wrt_candidates:
                ignore_overlaps = self.iou_calculator(
                    priors, gt_bboxes_ignore, mode='iof')
                ignore_max_overlaps, _ = ignore_overlaps.max(dim=1)
            else:
                ignore_overlaps = self.iou_calculator(
                    gt_bboxes_ignore, priors, mode='iof')
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

    def assign_wrt_overlaps(self, overlaps: Tensor,
                            gt_labels: Tensor) -> AssignResult:
        """Assign w.r.t. the overlaps of priors with gts.

        Args:
<<<<<<< HEAD:mmdet/core/bbox/assigners/max_iou_assigner.py
            overlaps (Tensor): k个gt与n个box之间的iou,[k, n].
            gt_labels (Tensor, optional): k个gt的label, [k, ].
=======
            overlaps (Tensor): Overlaps between k gt_bboxes and n bboxes,
                shape(k, n).
            gt_labels (Tensor): Labels of k gt_bboxes, shape (k, ).
>>>>>>> mmdetection/main:mmdet/models/task_modules/assigners/max_iou_assigner.py

        Returns:
            :obj:`AssignResult`: 分配结果.
        """
        num_gts, num_bboxes = overlaps.size(0), overlaps.size(1)
        # 步骤 1. 初始化样本值为-1(忽略样本).注:0代表负样本,正数代表分配给该box的gt索引
        assigned_gt_inds = overlaps.new_full((num_bboxes, ),
                                             -1,
                                             dtype=torch.long)

        if num_gts == 0 or num_bboxes == 0:
            # gt 或 box数量为0,则返回一个空的AssignResult
            max_overlaps = overlaps.new_zeros((num_bboxes, ))
            assigned_labels = overlaps.new_full((num_bboxes, ),
                                                -1,
                                                dtype=torch.long)
            if num_gts == 0:
                # 没有gt,所有anchor都更新为负样本
                assigned_gt_inds[:] = 0
            return AssignResult(
                num_gts=num_gts,
                gt_inds=assigned_gt_inds,
                max_overlaps=max_overlaps,
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
            # 例如,如果 prior_a 与 gt_1 和 gt_2 有 0.9 和 0.8 的iou,则 gt_1 将在步骤 3 中被指定为 prior_a 的最佳GT
            # 但是,如果 gt_2 的最佳prior是prior_a,则 prior_a 的最佳GT将被gt_2覆盖
            # 这可能是它没有在 ROI Heads 中使用的原因.
            for i in range(num_gts):
                if gt_max_overlaps[i] >= self.min_pos_iou:
                    if self.gt_max_assign_all:
                        max_iou_inds = overlaps[i, :] == gt_max_overlaps[i]
                        assigned_gt_inds[max_iou_inds] = i + 1
                    else:
                        assigned_gt_inds[gt_argmax_overlaps[i]] = i + 1

<<<<<<< HEAD:mmdet/core/bbox/assigners/max_iou_assigner.py
        if gt_labels is not None:  # 由于负样本在assigned_gt_inds上记为0,所以只能通过>0的值得gt索引来获取gt label
            assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), -1)
            pos_inds = torch.nonzero(
                assigned_gt_inds > 0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None
=======
        assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), -1)
        pos_inds = torch.nonzero(
            assigned_gt_inds > 0, as_tuple=False).squeeze()
        if pos_inds.numel() > 0:
            assigned_labels[pos_inds] = gt_labels[assigned_gt_inds[pos_inds] -
                                                  1]
>>>>>>> mmdetection/main:mmdet/models/task_modules/assigners/max_iou_assigner.py

        # assigned_gt_inds代表所有box对应的样本属性[-1,len(gt)],其中-1为忽略,0为负样本,其余为gt index(1-base)
        # max_overlaps代表单个anchor与所有gt的最大iou,shape为(n,), n为anchor个数
        # assigned_labels 代表所有box对应的样本属性[-1,len(class)-1],其中-1为负样本,其余为gt label
        return AssignResult(
            num_gts=num_gts,
            gt_inds=assigned_gt_inds,
            max_overlaps=max_overlaps,
            labels=assigned_labels)
