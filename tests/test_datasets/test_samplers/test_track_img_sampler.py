# Copyright (c) OpenMMLab. All rights reserved.
from collections.abc import Iterable
from copy import deepcopy
from unittest import TestCase

from mmengine.dataset import ClassBalancedDataset, ConcatDataset

from mmdet.datasets import MOTChallengeDataset, TrackImgSampler


class TestTrackImgSampler(TestCase):

    def test_iter_base_video_dataset(self):
        # train mode
        dataset = MOTChallengeDataset(
            data_prefix=dict(img_path='imgs'),
            ann_file='tests/data/mot_sample.json',
            metainfo=dict(classes=('pedestrian')),
            filter_cfg=dict(filter_empty_gt=True, min_size=32),
            test_mode=False,
            pipeline=[])
        video_sampler = TrackImgSampler(dataset)
        assert len(video_sampler) == 5
        iterator = iter(video_sampler)
        assert isinstance(iterator, Iterable)
        for index in iterator:
            assert isinstance(index, tuple)
            video_index, frame_index = index
            assert video_index < 2
            if video_index == 0:
                assert frame_index >= 0 and frame_index < 3
            else:
                assert frame_index >= 0 and frame_index < 2

        # test mode
        dataset = MOTChallengeDataset(
            data_prefix=dict(img_path='imgs'),
            ann_file='tests/data/mot_sample.json',
            metainfo=dict(classes=('pedestrian')),
            filter_cfg=dict(filter_empty_gt=True, min_size=32),
            test_mode=True,
            pipeline=[])
        video_sampler = TrackImgSampler(dataset)
        assert len(video_sampler) == 5
        assert len(video_sampler.indices) == 1

    def test_iter_concat_dataset(self):
        single_dataset = MOTChallengeDataset(
            data_prefix=dict(img_path='imgs'),
            ann_file='tests/data/mot_sample.json',
            metainfo=dict(classes=('pedestrian')),
            filter_cfg=dict(filter_empty_gt=True, min_size=32),
            test_mode=False,
            pipeline=[])

        dataset = ConcatDataset([single_dataset, deepcopy(single_dataset)])
        video_sampler = TrackImgSampler(dataset)
        assert len(video_sampler) == 10
        iterator = iter(video_sampler)
        assert isinstance(iterator, Iterable)
        for index in iterator:
            assert isinstance(index, tuple)
            video_index, frame_index = index
            assert video_index < 4
            if video_index == 0:
                assert frame_index >= 0 and frame_index < 3
            elif video_index == 3:
                assert frame_index >= 0 and frame_index < 2

    def test_iter_class_balanced_dataset(self):
        single_dataset = MOTChallengeDataset(
            data_prefix=dict(img_path='imgs'),
            ann_file='tests/data/mot_sample.json',
            metainfo=dict(classes=('pedestrian', 'person_on_vehicle')),
            filter_cfg=dict(filter_empty_gt=True, min_size=32),
            visibility_thr=0.1,
            test_mode=False,
            pipeline=[])

        dataset = ClassBalancedDataset(single_dataset, oversample_thr=0.6)
        video_sampler = TrackImgSampler(dataset)
        assert len(video_sampler) == 8
        iterator = iter(video_sampler)
        assert isinstance(iterator, Iterable)
        for index in iterator:
            assert isinstance(index, tuple)
            video_index, frame_index = index
            assert video_index < 3
            if video_index == 0 or video_index == 2:
                assert frame_index >= 0 and frame_index < 3
            else:
                assert frame_index >= 0 and frame_index < 2
