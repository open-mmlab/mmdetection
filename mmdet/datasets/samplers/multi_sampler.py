# Copyright (c) OpenMMLab. All rights reserved.
import itertools

import numpy as np
import torch
from mmcv.runner import get_dist_info
from torch.utils.data import Sampler

from mmdet.core.utils import sync_random_seed


class MultiSourceInfiniteSampler(Sampler):
    r"""Multi-Source Infinite Sampler.

        According to the sampling ratio, sample data from different
        datasets to form batches for ``IterBasedRunner``.

    Args:
        dataset: Dataset used for sampling.
        sample_ratio (list[int | float]): The sampling ratio of different
            datasets in a batch.
        samples_per_gpu (int): When model is :obj:`DistributedDataParallel`,
            it is the number of training samples on each GPU.
            When model is :obj:`DataParallel`, it is
            `num_gpus * samples_per_gpu`.
            Default : 1.
        num_replicas (int, optional): Number of processes participating in
            distributed training.
        rank (int, optional): Rank of the current process
            within ``num_replicas``.
        seed (int, optional): random seed used to shuffle the sampler if
            ``shuffle=True``. This number should be identical across all
            processes in the distributed group. Default: 0.
        shuffle (bool): Whether shuffle the dataset or not. Default: True.
    """

    def __init__(self,
                 dataset,
                 sample_ratio,
                 samples_per_gpu=1,
                 num_replicas=None,
                 rank=None,
                 seed=0,
                 shuffle=True):
        _rank, _num_replicas = get_dist_info()
        if num_replicas is None:
            num_replicas = _num_replicas
        if rank is None:
            rank = _rank

        self.rank = rank
        self.num_replicas = num_replicas

        assert hasattr(dataset, 'cumulative_sizes')
        self.dataset = dataset
        self.cumulative_sizes = [0] + dataset.cumulative_sizes
        assert isinstance(sample_ratio, list)
        assert len(sample_ratio) == len(dataset.cumulative_sizes)
        self.sample_ratio = sample_ratio
        self.samples_per_gpu = samples_per_gpu
        self.sample_num = [
            int(samples_per_gpu * sr / sum(sample_ratio))
            for sr in sample_ratio
        ]
        self.sample_num[0] = samples_per_gpu - sum(self.sample_num[1:])
        assert sum(self.sample_num) == self.samples_per_gpu, \
            f'The sum of sample_num must be equal to ' \
            f'samples_per_gpu, but get {self.sample_num}'

        self.seed = sync_random_seed(seed)
        self.shuffle = shuffle
        self.indices_per_source = {
            source: self._indices_of_rank(len(ds))
            for source, ds in enumerate(dataset.datasets)
        }

    def _infinite_indices(self, sample_size):
        """Infinitely yield a sequence of indices."""
        g = torch.Generator()
        g.manual_seed(self.seed)
        while True:
            if self.shuffle:
                yield from torch.randperm(sample_size, generator=g).tolist()
            else:
                yield from torch.arange(sample_size).tolist()

    def _indices_of_rank(self, sample_size):
        """Slice the infinite indices by rank."""
        yield from itertools.islice(
            self._infinite_indices(sample_size), self.rank, None,
            self.num_replicas)

    def __iter__(self):
        batch_buffer = []
        while True:
            for source, num in enumerate(self.sample_num):
                batch_buffer_per_source = []
                for idx in self.indices_per_source[source]:
                    idx += self.cumulative_sizes[source]
                    batch_buffer_per_source.append(idx)
                    if len(batch_buffer_per_source) == num:
                        batch_buffer += batch_buffer_per_source
                        break
            yield from batch_buffer
            batch_buffer = []

    def __len__(self):
        return len(self.dataset)

    def set_epoch(self, epoch):
        """Not supported in `IterationBased` runner."""
        raise NotImplementedError


class GroupMultiSourceInfiniteSampler(MultiSourceInfiniteSampler):
    r"""Group Multi-Source Infinite Sampler.

        According to the sampling ratio, sample data from different
        datasets but the same group to form batches for ``IterBasedRunner``.

    Args:
        dataset: Dataset used for sampling.
        sample_ratio (list[int | float]): The sampling ratio of different
            datasets in a batch.
        samples_per_gpu (int): When model is :obj:`DistributedDataParallel`,
            it is the number of training samples on each GPU.
            When model is :obj:`DataParallel`, it is
            `num_gpus * samples_per_gpu`.
            Default : 1.
        num_replicas (int, optional): Number of processes participating in
            distributed training.
        rank (int, optional): Rank of the current process
            within ``num_replicas``.
        seed (int, optional): random seed used to shuffle the sampler if
            ``shuffle=True``. This number should be identical across all
            processes in the distributed group. Default: 0.
        shuffle (bool): Whether shuffle the indices of a dummy ``epoch``
            for different datasets, it should be noted that ``shuffle``
            can not guarantee that you can generate sequential indices
            because it need to ensure that all indices in a batch is
            in a group. Default: True.

    """

    def __init__(self,
                 dataset,
                 sample_ratio,
                 samples_per_gpu=1,
                 num_replicas=None,
                 rank=None,
                 seed=0,
                 shuffle=True):
        super(GroupMultiSourceInfiniteSampler, self).__init__(
            dataset=dataset,
            sample_ratio=sample_ratio,
            samples_per_gpu=samples_per_gpu,
            num_replicas=num_replicas,
            rank=rank,
            seed=seed,
            shuffle=shuffle)

        assert hasattr(dataset, 'flag')
        self.group_sizes = np.bincount(self.dataset.flag)
        self.group_ratio = self.group_sizes / sum(self.group_sizes)

        self.indices_source_group = {
            group: {
                source: np.where(ds.flag == group)[0]
                for source, ds in enumerate(dataset.datasets)
            }
            for group in range(len(self.group_ratio))
        }

        self.indices_per_source_per_group = {
            group: {
                source: self._indices_of_rank(sum(ds.flag == group))
                for source, ds in enumerate(dataset.datasets)
            }
            for group in range(len(self.group_ratio))
        }

    def __iter__(self):
        batch_buffer = []
        while True:
            group = np.random.choice(
                list(range(len(self.group_ratio))), p=self.group_ratio)
            for source, num in enumerate(self.sample_num):
                batch_buffer_per_source_per_group = []
                for idx in self.indices_per_source_per_group[group][source]:
                    idx = self.indices_source_group[group][source][
                        idx] + self.cumulative_sizes[source]
                    batch_buffer_per_source_per_group.append(idx)
                    if len(batch_buffer_per_source_per_group) == num:
                        batch_buffer += batch_buffer_per_source_per_group
                        break
            yield from batch_buffer
            batch_buffer = []
