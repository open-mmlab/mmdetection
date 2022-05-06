# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence

from torch.utils.data import BatchSampler, Sampler

from mmdet.registry import DATA_SAMPLERS


# TODO: maybe replace with a data_loader wrapper
@DATA_SAMPLERS.register_module()
class AspectRatioBatchSampler(BatchSampler):
    """A sampler wrapper for grouping images with similar aspect ratio (< 1 or.

    >= 1) into a same batch.

    Args:
        sampler (Sampler): Base sampler.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``.
    """

    def __init__(self,
                 sampler: Sampler,
                 batch_size: int,
                 drop_last: bool = False) -> None:
        if not isinstance(sampler, Sampler):
            raise TypeError('sampler should be an instance of ``Sampler``, '
                            f'but got {sampler}')
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError('batch_size should be a positive integer value, '
                             f'but got batch_size={batch_size}')
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        # two groups for w < h and w >= h
        self._aspect_ratio_buckets = [[] for _ in range(2)]

    def __iter__(self) -> Sequence[int]:
        for idx in self.sampler:
            data_info = self.sampler.dataset.get_data_info(idx)
            width, height = data_info['width'], data_info['height']
            bucket_id = 0 if width < height else 1
            bucket = self._aspect_ratio_buckets[bucket_id]
            bucket.append(idx)
            # yield a batch of indices in the same aspect ratio group
            if len(bucket) == self.batch_size:
                yield bucket[:]
                del bucket[:]

        # yield the rest data and reset the bucket
        left_data = self._aspect_ratio_buckets[0] + self._aspect_ratio_buckets[
            1]
        self._aspect_ratio_buckets = [[] for _ in range(2)]
        while len(left_data) > 0:
            if len(left_data) <= self.batch_size:
                if not self.drop_last:
                    yield left_data[:]
                left_data = []
            else:
                yield left_data[:self.batch_size]
                left_data = left_data[self.batch_size:]

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size
