# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import TASK_UTILS
from .base_sampler import BaseSampler
from .sampling_result import SamplingResult


@TASK_UTILS.register_module()
class PseudoSampler(BaseSampler):
    """A pseudo sampler that does not do sampling actually."""

    def __init__(self, **kwargs):
        pass

    def _sample_pos(self, **kwargs):
        """Sample positive samples."""
        raise NotImplementedError

    def _sample_neg(self, **kwargs):
        """Sample negative samples."""
        raise NotImplementedError

    def sample(self, assign_result: SamplingResult, *args, **kwargs):
        """Directly returns the positive and negative indices  of samples.

        Args:
            assign_result (:obj:`SamplingResult`): Bbox assigning results.

        Returns:
            :obj:`SamplingResult`: sampler results
        """
        return assign_result
