# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence

from torch.utils.data import BatchSampler, Sampler
from mmdet.registry import DATA_SAMPLERS
import numpy as np


@DATA_SAMPLERS.register_module()
class MultiTaskAspectRatioBatchSampler(BatchSampler):
    def __init__(self,
                 sampler: Sampler,
                 batch_size: int,
                 drop_last: bool = True,
                 od_to_rec_prob=0.7) -> None:
        if not isinstance(sampler, Sampler):
            raise TypeError('sampler should be an instance of ``Sampler``, '
                            f'but got {sampler}')
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError('batch_size should be a positive integer value, '
                             f'but got batch_size={batch_size}')
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        # two groups for w < h and w >= h and two task
        self._aspect_ratio_buckets = [[] for _ in range(2 * 2)]
        self.od_to_rec_prob = od_to_rec_prob

    def __iter__(self) -> Sequence[int]:
        for idx in self.sampler:
            data_info = self.sampler.dataset.get_data_info(idx)
            width, height = data_info['width'], data_info['height']
            bucket_id = 0 if width < height else 1

            if data_info['dataset_mode'] == 'OD':
                if np.random.random() > 1-self.od_to_rec_prob:
                    data_info['dataset_mode'] = 'REC'

            # REC: 0 2
            # VG and OD: 1 3
            if data_info['dataset_mode'] == 'REC':
                bucket_id = bucket_id * 2
            else:
                bucket_id = bucket_id * 2 + 1
            bucket = self._aspect_ratio_buckets[bucket_id]
            bucket.append(idx)
            # yield a batch of indices in the same aspect ratio group
            if len(bucket) == self.batch_size:
                yield bucket[:]
                del bucket[:]

        # yield the rest data and reset the bucket
        # left_data = self._aspect_ratio_buckets[0] + self._aspect_ratio_buckets[
        #     1] + self._aspect_ratio_buckets[2] + self._aspect_ratio_buckets[3]
        self._aspect_ratio_buckets = [[] for _ in range(2 * 2)]
        # while len(left_data) > 0:
        #     if len(left_data) <= self.batch_size:
        #         if not self.drop_last:
        #             yield left_data[:]
        #         left_data = []
        #     else:
        #         yield left_data[:self.batch_size]
        #         left_data = left_data[self.batch_size:]

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size
