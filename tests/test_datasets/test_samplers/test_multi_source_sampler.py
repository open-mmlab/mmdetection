# Copyright (c) OpenMMLab. All rights reserved.

import bisect
from unittest import TestCase
from unittest.mock import patch

import numpy as np
from torch.utils.data import ConcatDataset, Dataset

from mmdet.datasets.samplers import GroupMultiSourceSampler, MultiSourceSampler


class DummyDataset(Dataset):

    def __init__(self, length, flag):
        self.length = length
        self.flag = flag
        self.shapes = np.random.random((length, 2))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.shapes[idx]

    def get_data_info(self, idx):
        return dict(
            width=self.shapes[idx][0],
            height=self.shapes[idx][1],
            flag=self.flag)


class DummyConcatDataset(ConcatDataset):

    def _get_ori_dataset_idx(self, idx):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        sample_idx = idx if dataset_idx == 0 else idx - self.cumulative_sizes[
            dataset_idx - 1]
        return dataset_idx, sample_idx

    def get_data_info(self, idx: int):
        dataset_idx, sample_idx = self._get_ori_dataset_idx(idx)
        return self.datasets[dataset_idx].get_data_info(sample_idx)


class TestMultiSourceSampler(TestCase):

    @patch('mmengine.dist.get_dist_info', return_value=(7, 8))
    def setUp(self, mock):
        self.length_a = 100
        self.dataset_a = DummyDataset(self.length_a, flag='a')
        self.length_b = 1000
        self.dataset_b = DummyDataset(self.length_b, flag='b')
        self.dataset = DummyConcatDataset([self.dataset_a, self.dataset_b])

    def test_multi_source_sampler(self):
        # test dataset is not ConcatDataset
        with self.assertRaises(AssertionError):
            MultiSourceSampler(
                self.dataset_a, batch_size=5, source_ratio=[1, 4])
        # test invalid batch_size
        with self.assertRaises(AssertionError):
            MultiSourceSampler(
                self.dataset_a, batch_size=-5, source_ratio=[1, 4])
        # test source_ratio longer then dataset
        with self.assertRaises(AssertionError):
            MultiSourceSampler(
                self.dataset, batch_size=5, source_ratio=[1, 2, 4])
        sampler = MultiSourceSampler(
            self.dataset, batch_size=5, source_ratio=[1, 4])
        sampler = iter(sampler)
        flags = []
        for i in range(100):
            idx = next(sampler)
            flags.append(self.dataset.get_data_info(idx)['flag'])
        flags_gt = ['a', 'b', 'b', 'b', 'b'] * 20
        self.assertEqual(flags, flags_gt)


class TestGroupMultiSourceSampler(TestCase):

    @patch('mmengine.dist.get_dist_info', return_value=(7, 8))
    def setUp(self, mock):
        self.length_a = 100
        self.dataset_a = DummyDataset(self.length_a, flag='a')
        self.length_b = 1000
        self.dataset_b = DummyDataset(self.length_b, flag='b')
        self.dataset = DummyConcatDataset([self.dataset_a, self.dataset_b])

    def test_group_multi_source_sampler(self):
        sampler = GroupMultiSourceSampler(
            self.dataset, batch_size=5, source_ratio=[1, 4])
        sampler = iter(sampler)
        flags = []
        groups = []
        for i in range(100):
            idx = next(sampler)
            data_info = self.dataset.get_data_info(idx)
            flags.append(data_info['flag'])
            group = 0 if data_info['width'] < data_info['height'] else 1
            groups.append(group)
        flags_gt = ['a', 'b', 'b', 'b', 'b'] * 20
        self.assertEqual(flags, flags_gt)
        groups = set(
            [sum(x) for x in (groups[k:k + 5] for k in range(0, 100, 5))])
        groups_gt = set([0, 5])
        self.assertEqual(groups, groups_gt)
