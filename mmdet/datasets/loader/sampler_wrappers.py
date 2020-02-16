import math
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Sampler

from mmdet.utils import build_from_cfg
from ..registry import SAMPLERS


class PseudoDataset(object):
    """
    Pseudo Dataset with flag
    Args:
        dataset (:obj:`Dataset`): The dataset to be sample.
        indices (nd.array): Map new sample indices to origin indices,
                        indices[i] is the new sampled index of dataset.
                        e.g. dataset[indices[i]]
    """

    def __init__(self, dataset, indices):
        self.indices = indices
        if hasattr(dataset, 'flag'):
            self.flag = np.zeros(len(indices), dtype=np.uint8)
            for idx, ori_idx in enumerate(indices):
                self.flag[idx] = dataset.flag[ori_idx]

    def __len__(self):
        return len(self.indices)


class BaseSampler(Sampler):

    def __init__(self, parent_sampler):
        self.parent_sampler = parent_sampler
        self.dataset = parent_sampler.dataset
        if hasattr(parent_sampler, 'epoch'):
            self.epoch = parent_sampler.epoch

    def set_epoch(self, epoch=None):
        if hasattr(self, 'epoch'):
            if epoch is None:
                epoch = self.epoch
            self.epoch = epoch
            self.parent_sampler.epoch = epoch

    def _get_sample_indices(self):
        sample_indices = np.arange(len(self.dataset))
        return sample_indices

    def __iter__(self):
        sample_indices = self._get_sample_indices()
        self.parent_sampler.reset_dataset(
            PseudoDataset(self.dataset, sample_indices))
        indices = sample_indices[list(self.parent_sampler)].tolist()
        assert len(indices) == len(self), '{} vs {}'.format(
            len(indices), len(self))
        return iter(indices)

    def __len__(self):
        return len(self.parent_sampler)


@SAMPLERS.register_module
class RepeatSampler(BaseSampler):
    """A wrapper of repeated sampler.

    The length of repeated sampler will be `times` larger than the original
    dataset. This is useful when the data loading time is long but the dataset
    is small. Using RepeatSampler can reduce the data loading time between
    epochs.

    Args:
        times (int): Repeat times.
    """

    def __init__(self, times, *args, **kwargs):
        super(RepeatSampler, self).__init__(*args, **kwargs)
        self.times = times

    def _get_sample_indices(self):
        ori_length = len(self.dataset)
        repeat_length = self.times * len(self.dataset)
        sample_indices = np.array(
            [idx % ori_length for idx in range(repeat_length)])
        return sample_indices


# Modified from https://github.com/facebookresearch/detectron2/blob/41d475b75a230221e21d9cac5d69655e3415e3a4/detectron2/data/samplers/distributed_sampler.py#L57 # noqa
@SAMPLERS.register_module
class RepeatFactorSampler(BaseSampler):
    """
    Similar to RepeatSampler, but suitable for training on class imbalanced
    datasets like LVIS. In each epoch, an image may appear multiple times
    based on its "repeat factor". The repeat factor for an image is a function
    of the frequency the rarest category labeled in that image. The "frequency
    of category c" in [0, 1] is defined as the fraction of images in the
    training set (without repeats) in which category c appears.

    Args:
        repeat_thresh (float): frequency threshold to repeat.
    """

    def __init__(self, repeat_thr, *args, **kwargs):
        super(RepeatFactorSampler, self).__init__(*args, **kwargs)
        self.repeat_thr = repeat_thr
        repeat_factors = self._get_repeat_factors(self.dataset, repeat_thr)

        self._int_part = torch.trunc(repeat_factors)
        self._frac_part = repeat_factors - self._int_part

        # init sampler length
        self._get_sample_indices()

    def _get_repeat_factors(self, dataset, repeat_thr):
        # 1. For each category c, compute the fraction # of images
        # that contain it: f(c)
        category_freq = defaultdict(int)
        for img_info in dataset.img_infos:  # For each image (without repeats)
            cat_ids = set(img_info['category_ids'])
            for cat_id in cat_ids:
                category_freq[cat_id] += 1
        num_images = len(dataset)
        for k, v in category_freq.items():
            category_freq[k] = v / num_images

        # 2. For each category c, compute the category-level repeat factor:
        #    r(c) = max(1, sqrt(t / f(c)))
        category_repeat = {
            cat_id: max(1.0, math.sqrt(repeat_thr / cat_freq))
            for cat_id, cat_freq in category_freq.items()
        }

        # 3. For each image I, compute the image-level repeat factor:
        #    r(I) = max_{c in I} r(c)
        repeat_factors = []
        for img_info in dataset.img_infos:
            cat_ids = set(img_info['category_ids'])
            repeat_factor = max(
                {category_repeat[cat_id]
                 for cat_id in cat_ids})
            repeat_factors.append(repeat_factor)

        return torch.Tensor(repeat_factors, dtype=torch.float32)

    def _get_sample_indices(self):
        """
        Create a list of dataset indices (with repeats) to use for one epoch.

        Returns:
            nd.array: list of sampler indices to use in one epoch. Each index
                is repeated based on its calculated repeat factor.
        """
        # Since repeat factors are fractional, we use stochastic rounding so
        # that the target repeat factor is achieved in expectation over the
        # course of training
        g = torch.Generator()
        g.manual_seed(self.epoch)
        rands = torch.rand(len(self._frac_part), generator=g)
        repeat_factors = self._int_part + (rands < self._frac_part).float()
        repeat_factors = repeat_factors.int().tolist()

        sample_indices = []
        for dataset_index, repeat_factor in enumerate(repeat_factors):
            sample_indices.extend([dataset_index] * repeat_factor)
        sample_indices = np.array(sample_indices)
        return sample_indices


def build_sampler(sampler_cfg, parent_sampler):
    sampler_cfg['parent_sampler'] = parent_sampler
    return build_from_cfg(sampler_cfg, SAMPLERS)
