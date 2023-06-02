# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import TASK_UTILS
from .base_sampler import BaseSampler


@TASK_UTILS.register_module()
class CombinedSampler(BaseSampler):
    """A sampler that combines positive sampler and negative sampler."""

    def __init__(self, pos_sampler, neg_sampler, **kwargs):
        super(CombinedSampler, self).__init__(**kwargs)
        self.pos_sampler = TASK_UTILS.build(pos_sampler, default_args=kwargs)
        self.neg_sampler = TASK_UTILS.build(neg_sampler, default_args=kwargs)

    def _sample_pos(self, **kwargs):
        """Sample positive samples."""
        raise NotImplementedError

    def _sample_neg(self, **kwargs):
        """Sample negative samples."""
        raise NotImplementedError
