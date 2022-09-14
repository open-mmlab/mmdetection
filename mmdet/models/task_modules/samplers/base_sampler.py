# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod

import torch
from mmengine.structures import InstanceData

from mmdet.structures.bbox import BaseBoxes, cat_boxes
from ..assigners import AssignResult
from .sampling_result import SamplingResult


class BaseSampler(metaclass=ABCMeta):
    """Base class of samplers.

    Args:
        num (int): Number of samples
        pos_fraction (float): Fraction of positive samples
        neg_pos_up (int): Upper bound number of negative and
            positive samples. Defaults to -1.
        add_gt_as_proposals (bool): Whether to add ground truth
            boxes as proposals. Defaults to True.
    """

    def __init__(self,
                 num: int,
                 pos_fraction: float,
                 neg_pos_ub: int = -1,
                 add_gt_as_proposals: bool = True,
                 **kwargs) -> None:
        self.num = num
        self.pos_fraction = pos_fraction
        self.neg_pos_ub = neg_pos_ub
        self.add_gt_as_proposals = add_gt_as_proposals
        self.pos_sampler = self
        self.neg_sampler = self

    @abstractmethod
    def _sample_pos(self, assign_result: AssignResult, num_expected: int,
                    **kwargs):
        """Sample positive samples."""
        pass

    @abstractmethod
    def _sample_neg(self, assign_result: AssignResult, num_expected: int,
                    **kwargs):
        """Sample negative samples."""
        pass

    def sample(self, assign_result: AssignResult, pred_instances: InstanceData,
               gt_instances: InstanceData, **kwargs) -> SamplingResult:
        """Sample positive and negative bboxes.

        This is a simple implementation of bbox sampling given candidates,
        assigning results and ground truth bboxes.

        Args:
            assign_result (:obj:`AssignResult`): Assigning results.
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

        Returns:
            :obj:`SamplingResult`: Sampling result.

        Example:
            >>> from mmengine.structures import InstanceData
            >>> from mmdet.models.task_modules.samplers import RandomSampler,
            >>> from mmdet.models.task_modules.assigners import AssignResult
            >>> from mmdet.models.task_modules.samplers.
            ... sampling_result import ensure_rng, random_boxes
            >>> rng = ensure_rng(None)
            >>> assign_result = AssignResult.random(rng=rng)
            >>> pred_instances = InstanceData()
            >>> pred_instances.priors = random_boxes(assign_result.num_preds,
            ...                                      rng=rng)
            >>> gt_instances = InstanceData()
            >>> gt_instances.bboxes = random_boxes(assign_result.num_gts,
            ...                                    rng=rng)
            >>> gt_instances.labels = torch.randint(
            ...     0, 5, (assign_result.num_gts,), dtype=torch.long)
            >>> self = RandomSampler(num=32, pos_fraction=0.5, neg_pos_ub=-1,
            >>>                      add_gt_as_proposals=False)
            >>> self = self.sample(assign_result, pred_instances, gt_instances)
        """
        gt_bboxes = gt_instances.bboxes
        priors = pred_instances.priors
        gt_labels = gt_instances.labels
        if len(priors.shape) < 2:
            priors = priors[None, :]

        gt_flags = priors.new_zeros((priors.shape[0], ), dtype=torch.uint8)
        if self.add_gt_as_proposals and len(gt_bboxes) > 0:
            # When `gt_bboxes` and `priors` are all box type, convert
            # `gt_bboxes` type to `priors` type.
            if (isinstance(gt_bboxes, BaseBoxes)
                    and isinstance(priors, BaseBoxes)):
                gt_bboxes_ = gt_bboxes.convert_to(type(priors))
            else:
                gt_bboxes_ = gt_bboxes
            priors = cat_boxes([gt_bboxes_, priors], dim=0)
            assign_result.add_gt_(gt_labels)
            gt_ones = priors.new_ones(gt_bboxes_.shape[0], dtype=torch.uint8)
            gt_flags = torch.cat([gt_ones, gt_flags])

        num_expected_pos = int(self.num * self.pos_fraction)
        pos_inds = self.pos_sampler._sample_pos(
            assign_result, num_expected_pos, bboxes=priors, **kwargs)
        # We found that sampled indices have duplicated items occasionally.
        # (may be a bug of PyTorch)
        pos_inds = pos_inds.unique()
        num_sampled_pos = pos_inds.numel()
        num_expected_neg = self.num - num_sampled_pos
        if self.neg_pos_ub >= 0:
            _pos = max(1, num_sampled_pos)
            neg_upper_bound = int(self.neg_pos_ub * _pos)
            if num_expected_neg > neg_upper_bound:
                num_expected_neg = neg_upper_bound
        neg_inds = self.neg_sampler._sample_neg(
            assign_result, num_expected_neg, bboxes=priors, **kwargs)
        neg_inds = neg_inds.unique()

        sampling_result = SamplingResult(
            pos_inds=pos_inds,
            neg_inds=neg_inds,
            priors=priors,
            gt_bboxes=gt_bboxes,
            assign_result=assign_result,
            gt_flags=gt_flags)
        return sampling_result
