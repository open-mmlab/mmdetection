# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Dict, Iterator, Optional, Union

import numpy as np
import torch
from mmengine.dataset import BaseDataset
from mmengine.dist import get_dist_info, sync_random_seed
from torch.utils.data import Sampler

from mmdet.registry import DATA_SAMPLERS


@DATA_SAMPLERS.register_module()
class ClassAwareSampler(Sampler):
    r"""Sampler that restricts data loading to the label of the dataset.

    A class-aware sampling strategy to effectively tackle the
    non-uniform class distribution. The length of the training data is
    consistent with source data. Simple improvements based on `Relay
    Backpropagation for Effective Learning of Deep Convolutional
    Neural Networks <https://arxiv.org/abs/1512.05830>`_

    The implementation logic is referred to
    https://github.com/Sense-X/TSD/blob/master/mmdet/datasets/samplers/distributed_classaware_sampler.py

    Args:
        dataset: Dataset used for sampling.
        seed (int, optional): random seed used to shuffle the sampler.
            This number should be identical across all
            processes in the distributed group. Defaults to None.
        num_sample_class (int): The number of samples taken from each
            per-label list. Defaults to 1.
    """

    def __init__(self,
                 dataset: BaseDataset,
                 seed: Optional[int] = None,
                 num_sample_class: int = 1) -> None:
        rank, world_size = get_dist_info()
        self.rank = rank
        self.world_size = world_size

        self.dataset = dataset
        self.epoch = 0
        # Must be the same across all workers. If None, will use a
        # random seed shared among workers
        # (require synchronization among all workers)
        if seed is None:
            seed = sync_random_seed()
        self.seed = seed

        # The number of samples taken from each per-label list
        assert num_sample_class > 0 and isinstance(num_sample_class, int)
        self.num_sample_class = num_sample_class
        # Get per-label image list from dataset
        self.cat_dict = self.get_cat2imgs()

        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / world_size))
        self.total_size = self.num_samples * self.world_size

        # get number of images containing each category
        self.num_cat_imgs = [len(x) for x in self.cat_dict.values()]
        # filter labels without images
        self.valid_cat_inds = [
            i for i, length in enumerate(self.num_cat_imgs) if length != 0
        ]
        self.num_classes = len(self.valid_cat_inds)

    def get_cat2imgs(self) -> Dict[int, list]:
        """Get a dict with class as key and img_ids as values.

        Returns:
            dict[int, list]: A dict of per-label image list,
            the item of the dict indicates a label index,
            corresponds to the image index that contains the label.
        """
        classes = self.dataset.metainfo.get('classes', None)
        if classes is None:
            raise ValueError('dataset metainfo must contain `classes`')
        # sort the label index
        cat2imgs = {i: [] for i in range(len(classes))}
        for i in range(len(self.dataset)):
            cat_ids = set(self.dataset.get_cat_ids(i))
            for cat in cat_ids:
                cat2imgs[cat].append(i)
        return cat2imgs

    def __iter__(self) -> Iterator[int]:
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch + self.seed)

        # initialize label list
        label_iter_list = RandomCycleIter(self.valid_cat_inds, generator=g)
        # initialize each per-label image list
        data_iter_dict = dict()
        for i in self.valid_cat_inds:
            data_iter_dict[i] = RandomCycleIter(self.cat_dict[i], generator=g)

        def gen_cat_img_inds(cls_list, data_dict, num_sample_cls):
            """Traverse the categories and extract `num_sample_cls` image
            indexes of the corresponding categories one by one."""
            id_indices = []
            for _ in range(len(cls_list)):
                cls_idx = next(cls_list)
                for _ in range(num_sample_cls):
                    id = next(data_dict[cls_idx])
                    id_indices.append(id)
            return id_indices

        # deterministically shuffle based on epoch
        num_bins = int(
            math.ceil(self.total_size * 1.0 / self.num_classes /
                      self.num_sample_class))
        indices = []
        for i in range(num_bins):
            indices += gen_cat_img_inds(label_iter_list, data_iter_dict,
                                        self.num_sample_class)

        # fix extra samples to make it evenly divisible
        if len(indices) >= self.total_size:
            indices = indices[:self.total_size]
        else:
            indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        offset = self.num_samples * self.rank
        indices = indices[offset:offset + self.num_samples]
        assert len(indices) == self.num_samples

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


class RandomCycleIter:
    """Shuffle the list and do it again after the list have traversed.

    The implementation logic is referred to
    https://github.com/wutong16/DistributionBalancedLoss/blob/master/mllt/datasets/loader/sampler.py

    Example:
        >>> label_list = [0, 1, 2, 4, 5]
        >>> g = torch.Generator()
        >>> g.manual_seed(0)
        >>> label_iter_list = RandomCycleIter(label_list, generator=g)
        >>> index = next(label_iter_list)
    Args:
        data (list or ndarray): The data that needs to be shuffled.
        generator: An torch.Generator object, which is used in setting the seed
            for generating random numbers.
    """  # noqa: W605

    def __init__(self,
                 data: Union[list, np.ndarray],
                 generator: torch.Generator = None) -> None:
        self.data = data
        self.length = len(data)
        self.index = torch.randperm(self.length, generator=generator).numpy()
        self.i = 0
        self.generator = generator

    def __iter__(self) -> Iterator:
        return self

    def __len__(self) -> int:
        return len(self.data)

    def __next__(self):
        if self.i == self.length:
            self.index = torch.randperm(
                self.length, generator=self.generator).numpy()
            self.i = 0
        idx = self.data[self.index[self.i]]
        self.i += 1
        return idx
