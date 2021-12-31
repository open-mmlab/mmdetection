# Copyright (c) OpenMMLab. All rights reserved.
"""copy from
https://github.com/ZwwWayne/K-Net/blob/main/knet/det/mask_pseudo_sampler.py."""

import torch

from mmdet.core.bbox.builder import BBOX_SAMPLERS
from .base_sampler import BaseSampler
from .mask_sampling_result import MaskSamplingResult


@BBOX_SAMPLERS.register_module()
class MaskPseudoSampler(BaseSampler):
    """A pseudo sampler that does not do sampling actually."""

    def __init__(self, **kwargs):
        pass

    def _sample_pos(self, **kwargs):
        """Sample positive samples."""
        raise NotImplementedError

    def _sample_neg(self, **kwargs):
        """Sample negative samples."""
        raise NotImplementedError

    def sample(self, assign_result, masks, gt_masks, **kwargs):
        """Directly returns the positive and negative indices  of samples.

        Args:
            assign_result (:obj:`AssignResult`): Assigned results
            masks (torch.Tensor): Bounding boxes
            gt_masks (torch.Tensor): Ground truth boxes
        Returns:
            :obj:`SamplingResult`: sampler results
        """
        pos_inds = torch.nonzero(
            assign_result.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
        neg_inds = torch.nonzero(
            assign_result.gt_inds == 0, as_tuple=False).squeeze(-1).unique()
        gt_flags = masks.new_zeros(masks.shape[0], dtype=torch.uint8)
        sampling_result = MaskSamplingResult(pos_inds, neg_inds, masks,
                                             gt_masks, assign_result, gt_flags)
        return sampling_result
