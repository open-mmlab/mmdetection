# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence

from torch.utils.data import BatchSampler, Sampler

from mmdet.datasets.samplers.track_img_sampler import TrackImgSampler
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


@DATA_SAMPLERS.register_module()
class TrackAspectRatioBatchSampler(AspectRatioBatchSampler):
    """A sampler wrapper for grouping images with similar aspect ratio (< 1 or.

    >= 1) into a same batch.

    Args:
        sampler (Sampler): Base sampler.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``.
    """

    def __iter__(self) -> Sequence[int]:
        for idx in self.sampler:
            # hard code to solve TrackImgSampler
            if isinstance(self.sampler, TrackImgSampler):
                video_idx, _ = idx
            else:
                video_idx = idx
            # video_idx
            data_info = self.sampler.dataset.get_data_info(video_idx)
            # data_info {video_id, images, video_length}
            img_data_info = data_info['images'][0]
            width, height = img_data_info['width'], img_data_info['height']
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

    def __iter__(self) -> Sequence[int]:
        for idx in self.sampler:
            data_info = self.sampler.dataset.get_data_info(idx)
            width, height = data_info['width'], data_info['height']
            dataset_source_idx = self.sampler.dataset.get_dataset_source(idx)
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
        sizes = [0 for _ in range(self.num_datasets)]
        for idx in self.sampler:
            dataset_source_idx = self.sampler.dataset.get_dataset_source(idx)
            sizes[dataset_source_idx] += 1

        if self.drop_last:
            lens = 0
            for i in range(self.num_datasets):
                lens += sizes[i] // self.batch_size[i]
            return lens
        else:
            lens = 0
            for i in range(self.num_datasets):
                lens += (sizes[i] + self.batch_size[i] -
                         1) // self.batch_size[i]
            return lens
