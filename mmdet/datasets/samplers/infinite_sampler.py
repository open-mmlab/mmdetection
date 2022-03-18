# Copyright (c) OpenMMLab. All rights reserved.
import itertools

import numpy as np
import torch
from mmcv.runner import get_dist_info
from torch.utils.data.sampler import Sampler

from mmdet.core.utils import sync_random_seed


class InfiniteGroupBatchSampler(Sampler):
    """Similar to `BatchSampler` warping a `GroupSampler. It is designed for
    iteration-based runners like `IterBasedRunner` and yields a mini-batch
    indices each time, all indices in a batch should be in the same group.

    The implementation logic is referred to
    https://github.com/facebookresearch/detectron2/blob/main/detectron2/data/samplers/grouped_batch_sampler.py

    Args:
        dataset (object): The dataset.
        batch_size (int): When model is :obj:`DistributedDataParallel`,
            it is the number of training samples on each GPU.
            When model is :obj:`DataParallel`, it is
            `num_gpus * samples_per_gpu`.
            Default : 1.
        world_size (int, optional): Number of processes participating in
            distributed training. Default: None.
        rank (int, optional): Rank of current process. Default: None.
        seed (int): Random seed. Default: 0.
        shuffle (bool): Whether shuffle the indices of a dummy `epoch`, it
            should be noted that `shuffle` can not guarantee that you can
            generate sequential indices because it need to ensure
            that all indices in a batch is in a group. Default: True.
    """  # noqa: W605

    def __init__(self,
                 dataset,
                 batch_size=1,
                 world_size=None,
                 rank=None,
                 seed=0,
                 shuffle=True):
        _rank, _world_size = get_dist_info()
        if world_size is None:
            world_size = _world_size
        if rank is None:
            rank = _rank
        self.rank = rank
        self.world_size = world_size
        self.dataset = dataset
        self.batch_size = batch_size
        # In distributed sampling, different ranks should sample
        # non-overlapped data in the dataset. Therefore, this function
        # is used to make sure that each rank shuffles the data indices
        # in the same order based on the same seed. Then different ranks
        # could use different indices to select non-overlapped data from the
        # same data list.
        self.seed = sync_random_seed(seed)
        self.shuffle = shuffle

        assert hasattr(self.dataset, 'flag')
        self.flag = self.dataset.flag
        self.group_sizes = np.bincount(self.flag)
        # buffer used to save indices of each group
        self.buffer_per_group = {k: [] for k in range(len(self.group_sizes))}

        self.size = len(dataset)
        self.indices = self._indices_of_rank()

    def _infinite_indices(self):
        """Infinitely yield a sequence of indices."""
        g = torch.Generator()
        g.manual_seed(self.seed)
        while True:
            if self.shuffle:
                yield from torch.randperm(self.size, generator=g).tolist()

            else:
                yield from torch.arange(self.size).tolist()

    def _indices_of_rank(self):
        """Slice the infinite indices by rank."""
        yield from itertools.islice(self._infinite_indices(), self.rank, None,
                                    self.world_size)

    def __iter__(self):
        # once batch size is reached, yield the indices
        for idx in self.indices:
            flag = self.flag[idx]
            group_buffer = self.buffer_per_group[flag]
            group_buffer.append(idx)
            if len(group_buffer) == self.batch_size:
                yield group_buffer[:]
                del group_buffer[:]

    def __len__(self):
        """Length of base dataset."""
        return self.size

    def set_epoch(self, epoch):
        """Not supported in `IterationBased` runner."""
        raise NotImplementedError


class InfiniteBatchSampler(Sampler):
    """Similar to `BatchSampler` warping a `DistributedSampler. It is designed
    iteration-based runners like `IterBasedRunner` and yields a mini-batch
    indices each time.

    The implementation logic is referred to
    https://github.com/facebookresearch/detectron2/blob/main/detectron2/data/samplers/grouped_batch_sampler.py

    Args:
        dataset (object): The dataset.
        batch_size (int): When model is :obj:`DistributedDataParallel`,
            it is the number of training samples on each GPU,
            When model is :obj:`DataParallel`, it is
            `num_gpus * samples_per_gpu`.
            Default : 1.
        world_size (int, optional): Number of processes participating in
            distributed training. Default: None.
        rank (int, optional): Rank of current process. Default: None.
        seed (int): Random seed. Default: 0.
        shuffle (bool): Whether shuffle the dataset or not. Default: True.
    """  # noqa: W605

    def __init__(self,
                 dataset,
                 batch_size=1,
                 world_size=None,
                 rank=None,
                 seed=0,
                 shuffle=True):
        _rank, _world_size = get_dist_info()
        if world_size is None:
            world_size = _world_size
        if rank is None:
            rank = _rank
        self.rank = rank
        self.world_size = world_size
        self.dataset = dataset
        self.batch_size = batch_size
        # In distributed sampling, different ranks should sample
        # non-overlapped data in the dataset. Therefore, this function
        # is used to make sure that each rank shuffles the data indices
        # in the same order based on the same seed. Then different ranks
        # could use different indices to select non-overlapped data from the
        # same data list.
        self.seed = sync_random_seed(seed)
        self.shuffle = shuffle
        self.size = len(dataset)
        self.indices = self._indices_of_rank()

    def _infinite_indices(self):
        """Infinitely yield a sequence of indices."""
        g = torch.Generator()
        g.manual_seed(self.seed)
        while True:
            if self.shuffle:
                yield from torch.randperm(self.size, generator=g).tolist()

            else:
                yield from torch.arange(self.size).tolist()

    def _indices_of_rank(self):
        """Slice the infinite indices by rank."""
        yield from itertools.islice(self._infinite_indices(), self.rank, None,
                                    self.world_size)

    def __iter__(self):
        # once batch size is reached, yield the indices
        batch_buffer = []
        for idx in self.indices:
            batch_buffer.append(idx)
            if len(batch_buffer) == self.batch_size:
                yield batch_buffer
                batch_buffer = []

    def __len__(self):
        """Length of base dataset."""
        return self.size

    def set_epoch(self, epoch):
        """Not supported in `IterationBased` runner."""
        raise NotImplementedError
