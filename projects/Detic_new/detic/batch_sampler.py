# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence

from torch.utils.data import BatchSampler, Sampler

from mmdet.registry import DATA_SAMPLERS


# TODO: maybe replace with a data_loader wrapper
@DATA_SAMPLERS.register_module()
class MultiDataAspectRatioBatchSampler(BatchSampler):
    """A sampler wrapper for grouping images with similar aspect ratio (< 1 or.

    >= 1) into a same batch for multi-source datasets.

    Args:
        sampler (Sampler): Base sampler.
        batch_size (Sequence(int)): Size of mini-batch for multi-source
        datasets.
        num_datasets(int): Number of multi-source datasets.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
        its size would be less than ``batch_size``.
    """

    def __init__(self,
                 sampler: Sampler,
                 batch_size: Sequence[int],
                 num_datasets: int,
                 drop_last: bool = True) -> None:
        if not isinstance(sampler, Sampler):
            raise TypeError('sampler should be an instance of ``Sampler``, '
                            f'but got {sampler}')
        self.sampler = sampler
        self.batch_size = batch_size
        self.num_datasets = num_datasets
        self.drop_last = drop_last
        # two groups for w < h and w >= h for each dataset --> 2 * num_datasets
        self._buckets = [[] for _ in range(2 * self.num_datasets)]

        sizes = [0 for _ in range(self.num_datasets)]
        self.it = list(self.sampler)
        for idx in self.it:
            data_info = self.sampler.dataset.get_data_info(idx)
            dataset_source_idx = data_info['dataset_source']
            sizes[dataset_source_idx] += 1
        self.sizes = sizes

    def __iter__(self) -> Sequence[int]:
        for idx in self.it:
            data_info = self.sampler.dataset.get_data_info(idx)
            width, height, dataset_source_idx = data_info['width'], data_info[
                'height'], data_info['dataset_source']
            aspect_ratio_bucket_id = 0 if width < height else 1
            bucket_id = dataset_source_idx * 2 + aspect_ratio_bucket_id
            bucket = self._buckets[bucket_id]
            bucket.append(idx)
            # yield a batch of indices in the same aspect ratio group
            if len(bucket) == self.batch_size[dataset_source_idx]:
                yield bucket[:]
                del bucket[:]

        # yield the rest data and reset the bucket
        for i in range(self.num_datasets):
            left_data = self._buckets[i * 2 + 0] + self._buckets[i * 2 + 1]
            while len(left_data) > 0:
                if len(left_data) <= self.batch_size[i]:
                    if not self.drop_last:
                        yield left_data[:]
                    left_data = []
                else:
                    yield left_data[:self.batch_size[i]]
                    left_data = left_data[self.batch_size[i]:]

        self._buckets = [[] for _ in range(2 * self.num_datasets)]

    def __len__(self) -> int:
        if self.drop_last:
            lens = 0
            for i in range(self.num_datasets):
                lens += self.sizes[i] // self.batch_size[i]
            return lens
        else:
            lens = 0
            for i in range(self.num_datasets):
                lens += (self.sizes[i] + self.batch_size[i] -
                         1) // self.batch_size[i]
            return lens
