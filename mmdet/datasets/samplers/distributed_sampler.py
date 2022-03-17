# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
from torch.utils.data import DistributedSampler as _DistributedSampler

from mmdet.core.utils import sync_random_seed


class DistributedSampler(_DistributedSampler):

    def __init__(self,
                 dataset,
                 num_replicas=None,
                 rank=None,
                 shuffle=True,
                 seed=0):
        super().__init__(
            dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)

        #  In distributed sampling, different ranks only need to
        #  sample some non-overlapping data in the dataset.
        #  It is necessary to synchronize the seeds of different
        #  ranks through `sync_random_seed` to ensure that the dataset
        #  indexes sampled by different ranks are exactly the same.
        self.seed = sync_random_seed(seed)

    def __iter__(self):
        # deterministically shuffle based on epoch
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch + self.seed)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        # in case that indices is shorter than half of total_size
        indices = (indices *
                   math.ceil(self.total_size / len(indices)))[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)
