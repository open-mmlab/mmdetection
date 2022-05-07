# Copyright (c) OpenMMLab. All rights reserved.
import pytest

from mmdet.datasets import CocoDataset


class TestCocoDataset:

    def test_coco_dataset(self):
        # test CocoDataset
        metainfo = dict(CLASSES=('bus', 'car'), task_name='new_task')
        dataset = CocoDataset(
            data_prefix=dict(img='imgs'),
            ann_file='tests/data/coco_sample.json',
            metainfo=metainfo,
            filter_cfg=dict(filter_empty_gt=True, min_size=32),
            pipeline=[])
        assert dataset.metainfo['CLASSES'] == ('bus', 'car')
        assert dataset.metainfo['task_name'] == 'new_task'

    def test_coco_dataset_without_filter_cfg(self):
        # test CocoDataset without filter_cfg
        dataset = CocoDataset(
            data_prefix=dict(img='imgs'),
            ann_file='tests/data/coco_sample.json',
            pipeline=[])
        assert len(dataset) == 2

    def test_coco_annotation_ids_unique(self):
        # test annotation ids not unique error
        metainfo = dict(CLASSES=('car', ), task_name='new_task')
        with pytest.raises(AssertionError):
            CocoDataset(
                data_prefix=dict(img='imgs'),
                ann_file='tests/data/coco_wrong_format_sample.json',
                metainfo=metainfo,
                pipeline=[])
