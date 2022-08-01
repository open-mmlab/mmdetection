# Copyright (c) OpenMMLab. All rights reserved.
import torch

from ..builder import BBOX_SAMPLERS
from .base_sampler import BaseSampler
from .sampling_result import SamplingResult


@BBOX_SAMPLERS.register_module()
class PseudoSampler(BaseSampler):
    """一个实际上不进行采样的伪采样器."""

    def __init__(self, **kwargs):
        pass

    def _sample_pos(self, **kwargs):
        """Sample positive samples."""
        raise NotImplementedError

    def _sample_neg(self, **kwargs):
        """Sample negative samples."""
        raise NotImplementedError

    def sample(self, assign_result, bboxes, gt_bboxes, *args, **kwargs):
        """直接返回样本的正负索引.

        Args:
            assign_result (:obj:`AssignResult`): box分配的结果
            bboxes (torch.Tensor): box
            gt_bboxes (torch.Tensor): gt

        Returns:
            :obj:`SamplingResult`: 采样的结果
        """
        # 在所有样本中正样本索引,(gt_ind 正数为gt索引,从1开始∈[1,len(gt)],0为负样本,-1为背景)
        pos_inds = torch.nonzero(
            assign_result.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
        # 在所有样本中负样本索引
        neg_inds = torch.nonzero(
            assign_result.gt_inds == 0, as_tuple=False).squeeze(-1).unique()
        gt_flags = bboxes.new_zeros(bboxes.shape[0], dtype=torch.uint8)
        sampling_result = SamplingResult(pos_inds, neg_inds, bboxes, gt_bboxes,
                                         assign_result, gt_flags)
        return sampling_result
