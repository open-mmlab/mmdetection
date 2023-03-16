# Copyright (c) OpenMMLab. All rights reserved.
import math
import random
from typing import Iterator, Optional, Sized

import numpy as np
from mmengine.dataset import ClassBalancedDataset, ConcatDataset
from mmengine.dist import get_dist_info, sync_random_seed
from torch.utils.data import Sampler

from mmdet.registry import DATA_SAMPLERS
from ..base_video_dataset import BaseVideoDataset


@DATA_SAMPLERS.register_module()
class TrackImgSampler(Sampler):
    """Sampler that providing image-level sampling outputs for video datasets
    in tracking tasks. It could be both used in both distributed and
    non-distributed environment.
    If using the default sampler in pytorch, the subsequent data receiver will
    get one video, which is not desired in some cases:
    (Take a non-distributed environment as an example)
    1. In test mode, we want only one image is fed into the data pipeline. This
    is in consideration of memory usage since feeding the whole video commonly
    requires a large amount of memory (>=20G on MOTChallenge17 dataset), which
    is not available in some machines.
    2. In training mode, we may want to make sure all the images in one video
    are randomly sampled once in one epoch and this can not be guaranteed in
    the default sampler in pytorch.

    Args:
        dataset (Sized): Dataset used for sampling.
        seed (int, optional): random seed used to shuffle the sampler. This
            number should be identical across all processes in the distributed
            group. Defaults to None.
    """

    def __init__(
        self,
        dataset: Sized,
        seed: Optional[int] = None,
    ) -> None:
        rank, world_size = get_dist_info()
        self.rank = rank
        self.world_size = world_size
        self.epoch = 0
        if seed is None:
            self.seed = sync_random_seed()
        else:
            self.seed = seed

        self.dataset = dataset
        self.indices = []
        # Hard code here to handle different dataset wrapper
        if isinstance(self.dataset, ConcatDataset):
            cat_datasets = self.dataset.datasets
            assert isinstance(
                cat_datasets[0], BaseVideoDataset
            ), f'expected BaseVideoDataset, but got {type(cat_datasets[0])}'
            self.test_mode = cat_datasets[0].test_mode
            assert not self.test_mode, "'ConcatDataset' should not exist in "
            'test mode'
            for dataset in cat_datasets:
                num_videos = len(dataset)
                for video_ind in range(num_videos):
                    self.indices.extend([
                        (video_ind, frame_ind) for frame_ind in range(
                            dataset.get_len_per_video(video_ind))
                    ])
        elif isinstance(self.dataset, ClassBalancedDataset):
            ori_dataset = self.dataset.dataset
            assert isinstance(
                ori_dataset, BaseVideoDataset
            ), f'expected BaseVideoDataset, but got {type(ori_dataset)}'
            self.test_mode = ori_dataset.test_mode
            assert not self.test_mode, "'ClassBalancedDataset' should not "
            'exist in test mode'
            video_indices = self.dataset.repeat_indices
            for index in video_indices:
                self.indices.extend([(index, frame_ind) for frame_ind in range(
                    ori_dataset.get_len_per_video(index))])
        else:
            assert isinstance(
                self.dataset, BaseVideoDataset
            ), 'TrackImgSampler is only supported in BaseVideoDataset or '
            'dataset wrapper: ClassBalancedDataset and ConcatDataset, but '
            f'got {type(self.dataset)} '
            self.test_mode = self.dataset.test_mode
            num_videos = len(self.dataset)

            if self.test_mode:
                # in test mode, the images belong to the same video must be put
                # on the same device.
                if num_videos < self.world_size:
                    raise ValueError(f'only {num_videos} videos loaded,'
                                     f'but {self.world_size} gpus were given.')
                chunks = np.array_split(
                    list(range(num_videos)), self.world_size)
                for videos_inds in chunks:
                    indices_chunk = []
                    for video_ind in videos_inds:
                        indices_chunk.extend([
                            (video_ind, frame_ind) for frame_ind in range(
                                self.dataset.get_len_per_video(video_ind))
                        ])
                    self.indices.append(indices_chunk)
            else:
                for video_ind in range(num_videos):
                    self.indices.extend([
                        (video_ind, frame_ind) for frame_ind in range(
                            self.dataset.get_len_per_video(video_ind))
                    ])

        if self.test_mode:
            self.num_samples = len(self.indices[self.rank])
            self.total_size = sum(
                [len(index_list) for index_list in self.indices])
        else:
            self.num_samples = int(
                math.ceil(len(self.indices) * 1.0 / self.world_size))
            self.total_size = self.num_samples * self.world_size

    def __iter__(self) -> Iterator:
        if self.test_mode:
            # in test mode, the order of frames can not be shuffled.
            indices = self.indices[self.rank]
        else:
            # deterministically shuffle based on epoch
            rng = random.Random(self.epoch + self.seed)
            indices = rng.sample(self.indices, len(self.indices))

            # add extra samples to make it evenly divisible
            indices += indices[:(self.total_size - len(indices))]
            assert len(indices) == self.total_size

            # subsample
            indices = indices[self.rank:self.total_size:self.world_size]
            assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
