import torch

from .base_sampler import BaseSampler
from .sampling_result import SamplingResult


class PseudoSampler(BaseSampler):

    def __init__(self, **kwargs):
        pass

    def _sample_pos(self, **kwargs):
        raise NotImplementedError

    def _sample_neg(self, **kwargs):
        raise NotImplementedError

    def sample(self, assign_result, bboxes, gt_bboxes, **kwargs):
        pos_inds = torch.nonzero(
            assign_result.gt_inds > 0).squeeze(-1).unique()
        neg_inds = torch.nonzero(
            assign_result.gt_inds == 0).squeeze(-1).unique()
        gt_flags = bboxes.new_zeros(bboxes.shape[0], dtype=torch.uint8)
        sampling_result = SamplingResult(pos_inds, neg_inds, bboxes, gt_bboxes,
                                         assign_result, gt_flags)
        return sampling_result
