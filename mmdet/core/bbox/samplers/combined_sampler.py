from ..builder import BBOX_SAMPLERS, build_sampler
from .base_sampler import BaseSampler


@BBOX_SAMPLERS.register_module()
class CombinedSampler(BaseSampler):
    """A sampler that combines positive sampler and negative sampler."""

    def __init__(self, pos_sampler, neg_sampler, **kwargs):
        super(CombinedSampler, self).__init__(**kwargs)
        self.pos_sampler = build_sampler(pos_sampler, **kwargs)
        self.neg_sampler = build_sampler(neg_sampler, **kwargs)

    def _sample_pos(self, **kwargs):
        """Sample positive samples."""
        raise NotImplementedError

    def _sample_neg(self, **kwargs):
        """Sample negative samples."""
        raise NotImplementedError
