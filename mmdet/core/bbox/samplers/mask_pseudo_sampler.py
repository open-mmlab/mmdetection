"""
copy from https://github.com/ZwwWayne/K-Net/blob/main/knet/det/mask_pseudo_sampler.py
"""

import torch

from .base_sampler import BaseSampler
from .sampling_result import SamplingResult
from mmdet.core.bbox.builder import BBOX_SAMPLERS


class MaskSamplingResult(SamplingResult):
    """Bbox sampling result.
    Example:
        >>> # xdoctest: +IGNORE_WANT
        >>> from mmdet.core.bbox.samplers.sampling_result import *  # NOQA
        >>> self = SamplingResult.random(rng=10)
        >>> print(f'self = {self}')
        self = <SamplingResult({
            'neg_masks': torch.Size([12, 4]),
            'neg_inds': tensor([ 0,  1,  2,  4,  5,  6,  7,  8,  9, 10, 11, 12]),
            'num_gts': 4,
            'pos_assigned_gt_inds': tensor([], dtype=torch.int64),
            'pos_masks': torch.Size([0, 4]),
            'pos_inds': tensor([], dtype=torch.int64),
            'pos_is_gt': tensor([], dtype=torch.uint8)
        })>
    """

    def __init__(self, pos_inds, neg_inds, masks, gt_masks, assign_result,
                 gt_flags):
        self.pos_inds = pos_inds
        self.neg_inds = neg_inds
        self.pos_masks = masks[pos_inds]
        self.neg_masks = masks[neg_inds]
        self.pos_is_gt = gt_flags[pos_inds]

        self.num_gts = gt_masks.shape[0]
        self.pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1

        if gt_masks.numel() == 0:
            # hack for index error case
            assert self.pos_assigned_gt_inds.numel() == 0
            self.pos_gt_masks = torch.empty_like(gt_masks)
        else:
            self.pos_gt_masks = gt_masks[self.pos_assigned_gt_inds, :]

        if assign_result.labels is not None:
            self.pos_gt_labels = assign_result.labels[pos_inds]
        else:
            self.pos_gt_labels = None

    @property
    def masks(self):
        """torch.Tensor: concatenated positive and negative boxes"""
        return torch.cat([self.pos_masks, self.neg_masks])

    def __nice__(self):
        data = self.info.copy()
        data['pos_masks'] = data.pop('pos_masks').shape
        data['neg_masks'] = data.pop('neg_masks').shape
        parts = [f"'{k}': {v!r}" for k, v in sorted(data.items())]
        body = '    ' + ',\n    '.join(parts)
        return '{\n' + body + '\n}'

    @property
    def info(self):
        """Returns a dictionary of info about the object."""
        return {
            'pos_inds': self.pos_inds,
            'neg_inds': self.neg_inds,
            'pos_masks': self.pos_masks,
            'neg_masks': self.neg_masks,
            'pos_is_gt': self.pos_is_gt,
            'num_gts': self.num_gts,
            'pos_assigned_gt_inds': self.pos_assigned_gt_inds,
        }


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