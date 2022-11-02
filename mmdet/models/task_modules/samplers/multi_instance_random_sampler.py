# Copyright (c) OpenMMLab. All rights reserved.
from typing import Union

import torch
from mmengine.structures import InstanceData
from numpy import ndarray
from torch import Tensor

from mmdet.registry import TASK_UTILS
from ..assigners import AssignResult
from .multi_instance_sampling_result import MultiInstanceSamplingResult
from .random_sampler import RandomSampler


@TASK_UTILS.register_module()
class MultiInsRandomSampler(RandomSampler):
    """Random sampler for multi instance.

    Note:
        Multi-instance means to predict multiple detection boxes with
        one proposal box. `AssignResult` may assign multiple gt boxes
        to each proposal box, in this case `RandomSampler` should be
        replaced by `MultiInsRandomSampler`
    """

    def _sample_pos(self, assign_result: AssignResult, num_expected: int,
                    **kwargs) -> Union[Tensor, ndarray]:
        """Randomly sample some positive samples.

        Args:
            assign_result (:obj:`AssignResult`): Bbox assigning results.
            num_expected (int): The number of expected positive samples

        Returns:
            Tensor or ndarray: sampled indices.
        """
        pos_inds = torch.nonzero(
            assign_result.labels[:, 0] > 0, as_tuple=False)
        if pos_inds.numel() != 0:
            pos_inds = pos_inds.squeeze(1)
        if pos_inds.numel() <= num_expected:
            return pos_inds
        else:
            return self.random_choice(pos_inds, num_expected)

    def _sample_neg(self, assign_result: AssignResult, num_expected: int,
                    **kwargs) -> Union[Tensor, ndarray]:
        """Randomly sample some negative samples.

        Args:
            assign_result (:obj:`AssignResult`): Bbox assigning results.
            num_expected (int): The number of expected positive samples

        Returns:
            Tensor or ndarray: sampled indices.
        """
        neg_inds = torch.nonzero(
            assign_result.labels[:, 0] == 0, as_tuple=False)
        if neg_inds.numel() != 0:
            neg_inds = neg_inds.squeeze(1)
        if len(neg_inds) <= num_expected:
            return neg_inds
        else:
            return self.random_choice(neg_inds, num_expected)

    def sample(self, assign_result: AssignResult, pred_instances: InstanceData,
               gt_instances: InstanceData,
               **kwargs) -> MultiInstanceSamplingResult:
        """Sample positive and negative bboxes.

        Args:
            assign_result (:obj:`AssignResult`): Assigning results from
                MultiInstanceAssigner.
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
            :obj:`MultiInstanceSamplingResult`: Sampling result.
        """

        assert 'batch_gt_instances_ignore' in kwargs, \
            'batch_gt_instances_ignore is necessary for MultiInsRandomSampler'

        gt_bboxes = gt_instances.bboxes
        ignore_bboxes = kwargs['batch_gt_instances_ignore'].bboxes
        gt_and_ignore_bboxes = torch.cat([gt_bboxes, ignore_bboxes], dim=0)
        priors = pred_instances.priors
        if len(priors.shape) < 2:
            priors = priors[None, :]
        priors = priors[:, :4]

        gt_flags = priors.new_zeros((priors.shape[0], ), dtype=torch.uint8)
        priors = torch.cat([priors, gt_and_ignore_bboxes], dim=0)
        gt_ones = priors.new_ones(
            gt_and_ignore_bboxes.shape[0], dtype=torch.uint8)
        gt_flags = torch.cat([gt_flags, gt_ones])

        num_expected_pos = int(self.num * self.pos_fraction)
        pos_inds = self.pos_sampler._sample_pos(assign_result,
                                                num_expected_pos)
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
        neg_inds = self.neg_sampler._sample_neg(assign_result,
                                                num_expected_neg)
        neg_inds = neg_inds.unique()

        sampling_result = MultiInstanceSamplingResult(
            pos_inds=pos_inds,
            neg_inds=neg_inds,
            priors=priors,
            gt_and_ignore_bboxes=gt_and_ignore_bboxes,
            assign_result=assign_result,
            gt_flags=gt_flags)
        return sampling_result
