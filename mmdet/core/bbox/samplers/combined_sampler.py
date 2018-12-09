from .random_sampler import RandomSampler
from ..assign_sampling import build_sampler


class CombinedSampler(RandomSampler):

    def __init__(self, num, pos_fraction, pos_sampler, neg_sampler, **kwargs):
        super(CombinedSampler, self).__init__(num, pos_fraction, **kwargs)
        default_args = dict(num=num, pos_fraction=pos_fraction)
        default_args.update(kwargs)
        self.pos_sampler = build_sampler(
            pos_sampler, default_args=default_args)
        self.neg_sampler = build_sampler(
            neg_sampler, default_args=default_args)
