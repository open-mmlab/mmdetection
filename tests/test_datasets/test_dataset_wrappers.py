# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from unittest.mock import MagicMock

import pytest
import torch
from mmengine.dataset import BaseDataset

from mmdet.datasets import SemiDataset


class TestSemiDataset:

    def setup(self):
        dataset = BaseDataset

        # create dataset_a
        data_info = dict(filename='color.jpg', height=288, width=512)
        dataset.parse_data_info = MagicMock(return_value=data_info)
        imgs = torch.rand((3, 32, 32))

        self.dataset_a = dataset(
            data_root=osp.join(osp.dirname(__file__), '../data/'),
            data_prefix=dict(img=''),
            ann_file='dummy_annotation.json')
        self.dataset_a.pipeline = MagicMock(return_value=dict(imgs=imgs))

        # create dataset_b
        data_info = dict(filename='gray.jpg', height=288, width=512)
        dataset.parse_data_info = MagicMock(return_value=data_info)
        imgs = torch.rand((3, 32, 32))
        self.dataset_b = dataset(
            data_root=osp.join(osp.dirname(__file__), '../data/'),
            data_prefix=dict(img=''),
            ann_file='dummy_annotation.json')
        self.dataset_b.pipeline = MagicMock(return_value=dict(imgs=imgs))
        # test init
        self.semi_dataset = SemiDataset(
            sup=self.dataset_a, unsup=self.dataset_b)

    def test_init(self):
        # Test build dataset from cfg.
        dataset_cfg_b = dict(
            type=BaseDataset,
            data_root=osp.join(osp.dirname(__file__), '../data/'),
            data_prefix=dict(img=''),
            ann_file='dummy_annotation.json')
        semi_dataset = SemiDataset(sup=self.dataset_a, unsup=dataset_cfg_b)
        semi_dataset.datasets[1].pipeline = self.dataset_b.pipeline
        assert len(semi_dataset) == len(self.semi_dataset)
        for i in range(len(semi_dataset)):
            assert (semi_dataset.get_data_info(i) ==
                    self.semi_dataset.get_data_info(i))
            assert (semi_dataset[i] == self.semi_dataset[i])

    def test_full_init(self):
        # test init with lazy_init=True
        self.semi_dataset.full_init()
        assert len(self.semi_dataset) == 6
        self.semi_dataset.full_init()
        self.semi_dataset._fully_initialized = False
        self.semi_dataset[1]
        assert len(self.semi_dataset) == 6

        with pytest.raises(NotImplementedError):
            self.semi_dataset.get_subset_(1)

        with pytest.raises(NotImplementedError):
            self.semi_dataset.get_subset(1)
        # Different meta information will raise error.
        with pytest.raises(ValueError):
            dataset_b = BaseDataset(
                data_root=osp.join(osp.dirname(__file__), '../data/'),
                data_prefix=dict(img=''),
                ann_file='dummy_annotation.json',
                metainfo=dict(classes=('cat')))
            SemiDataset(sup=self.dataset_a, unsup=dataset_b)

    def test_metainfo(self):
        assert self.semi_dataset.metainfo == self.dataset_a.metainfo

    def test_length(self):
        assert len(self.semi_dataset) == (
            len(self.dataset_a) + len(self.dataset_b))

    def test_getitem(self):
        assert (
            self.semi_dataset[0]['imgs'] == self.dataset_a[0]['imgs']).all()
        assert (self.semi_dataset[0]['imgs'] !=
                self.dataset_b[0]['imgs']).all()

        assert (
            self.semi_dataset[-1]['imgs'] == self.dataset_b[-1]['imgs']).all()
        assert (self.semi_dataset[-1]['imgs'] !=
                self.dataset_a[-1]['imgs']).all()

    def test_get_data_info(self):
        assert self.semi_dataset.get_data_info(
            0) == self.dataset_a.get_data_info(0)
        assert self.semi_dataset.get_data_info(
            0) != self.dataset_b.get_data_info(0)

        assert self.semi_dataset.get_data_info(
            -1) == self.dataset_b.get_data_info(-1)
        assert self.semi_dataset.get_data_info(
            -1) != self.dataset_a.get_data_info(-1)

    def test_get_ori_dataset_idx(self):
        assert self.semi_dataset._get_ori_dataset_idx(3) == (
            1, 3 - len(self.dataset_a))
        assert self.semi_dataset._get_ori_dataset_idx(-1) == (
            1, len(self.dataset_b) - 1)
        with pytest.raises(ValueError):
            assert self.semi_dataset._get_ori_dataset_idx(-10)
