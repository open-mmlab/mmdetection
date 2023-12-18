# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Iterator, Optional, Sequence, Sized

import torch
from mmengine.dist import get_dist_info, sync_random_seed
from torch.utils.data import Sampler

from mmdet.registry import DATA_SAMPLERS
from .class_aware_sampler import RandomCycleIter


@DATA_SAMPLERS.register_module()
class CustomSampleSizeSampler(Sampler):

    def __init__(self,
                 dataset: Sized,
                 dataset_size: Sequence[int],
                 ratio_mode: bool = False,
                 seed: Optional[int] = None,
                 round_up: bool = True) -> None:
        assert len(dataset.datasets) == len(dataset_size)
        rank, world_size = get_dist_info()
        self.rank = rank
        self.world_size = world_size

        self.dataset = dataset
        if seed is None:
            seed = sync_random_seed()
        self.seed = seed
        self.epoch = 0
        self.round_up = round_up

        total_size = 0
        total_size_fake = 0
        self.dataset_index = []
        self.dataset_cycle_iter = []
        new_dataset_size = []
        for dataset, size in zip(dataset.datasets, dataset_size):
            self.dataset_index.append(
                list(range(total_size_fake,
                           len(dataset) + total_size_fake)))
            total_size_fake += len(dataset)
            if size == -1:
                total_size += len(dataset)
                self.dataset_cycle_iter.append(None)
                new_dataset_size.append(-1)
            else:
                if ratio_mode:
                    size = int(size * len(dataset))
                assert size <= len(
                    dataset
                ), f'dataset size {size} is larger than ' \
                   f'dataset length {len(dataset)}'
                total_size += size
                new_dataset_size.append(size)

                g = torch.Generator()
                g.manual_seed(self.seed)
                self.dataset_cycle_iter.append(
                    RandomCycleIter(self.dataset_index[-1], generator=g))
        self.dataset_size = new_dataset_size

        if self.round_up:
            self.num_samples = math.ceil(total_size / world_size)
            self.total_size = self.num_samples * self.world_size
        else:
            self.num_samples = math.ceil((total_size - rank) / world_size)
            self.total_size = total_size

    def __iter__(self) -> Iterator[int]:
        """Iterate the indices."""
        # deterministically shuffle based on epoch and seed
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        out_index = []
        for data_size, data_index, cycle_iter in zip(self.dataset_size,
                                                     self.dataset_index,
                                                     self.dataset_cycle_iter):
            if data_size == -1:
                out_index += data_index
            else:
                index = [next(cycle_iter) for _ in range(data_size)]
                out_index += index

        index = torch.randperm(len(out_index), generator=g).numpy().tolist()
        indices = [out_index[i] for i in index]

        if self.round_up:
            indices = (
                indices *
                int(self.total_size / len(indices) + 1))[:self.total_size]
        indices = indices[self.rank:self.total_size:self.world_size]
        return iter(indices)

    def __len__(self) -> int:
        """The number of samples in this rank."""
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        """Sets the epoch for this sampler.

        When :attr:`shuffle=True`, this ensures all replicas use a different
        random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch
