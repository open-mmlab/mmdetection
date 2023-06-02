# Copyright (c) OpenMMLab. All rights reserved.
import unittest

from mmdet.datasets import Objects365V1Dataset, Objects365V2Dataset


class TestObjects365V1Dataset(unittest.TestCase):

    def test_obj365v1_dataset(self):
        # test Objects365V1Dataset
        metainfo = dict(classes=('bus', 'car'), task_name='new_task')
        dataset = Objects365V1Dataset(
            data_prefix=dict(img='imgs'),
            ann_file='tests/data/coco_sample.json',
            metainfo=metainfo,
            filter_cfg=dict(filter_empty_gt=True, min_size=32),
            pipeline=[],
            serialize_data=False,
            lazy_init=False)
        self.assertEqual(dataset.metainfo['classes'], ('bus', 'car'))
        self.assertEqual(dataset.metainfo['task_name'], 'new_task')
        self.assertListEqual(dataset.get_cat_ids(0), [0, 1])
        self.assertEqual(dataset.cat_ids, [1, 2])

    def test_obj365v1_with_unsorted_annotation(self):
        # test Objects365V1Dataset with unsorted annotations
        metainfo = dict(classes=('bus', 'car'), task_name='new_task')
        dataset = Objects365V1Dataset(
            data_prefix=dict(img='imgs'),
            ann_file='tests/data/Objects365/unsorted_obj365_sample.json',
            metainfo=metainfo,
            filter_cfg=dict(filter_empty_gt=True, min_size=32),
            pipeline=[],
            serialize_data=False,
            lazy_init=False)
        self.assertEqual(dataset.metainfo['classes'], ('bus', 'car'))
        self.assertEqual(dataset.metainfo['task_name'], 'new_task')
        # sort the unsorted annotations
        self.assertListEqual(dataset.get_cat_ids(0), [0, 1])
        self.assertEqual(dataset.cat_ids, [1, 2])

    def test_obj365v1_annotation_ids_unique(self):
        # test annotation ids not unique error
        metainfo = dict(classes=('car', ), task_name='new_task')
        with self.assertRaisesRegex(AssertionError, 'are not unique!'):
            Objects365V1Dataset(
                data_prefix=dict(img='imgs'),
                ann_file='tests/data/coco_wrong_format_sample.json',
                metainfo=metainfo,
                pipeline=[])


class TestObjects365V2Dataset(unittest.TestCase):

    def test_obj365v2_dataset(self):
        # test Objects365V2Dataset
        metainfo = dict(classes=('bus', 'car'), task_name='new_task')
        dataset = Objects365V2Dataset(
            data_prefix=dict(img='imgs'),
            ann_file='tests/data/coco_sample.json',
            metainfo=metainfo,
            filter_cfg=dict(filter_empty_gt=True, min_size=32),
            pipeline=[],
            serialize_data=False,
            lazy_init=False)
        self.assertEqual(dataset.metainfo['classes'], ('bus', 'car'))
        self.assertEqual(dataset.metainfo['task_name'], 'new_task')
        self.assertListEqual(dataset.get_cat_ids(0), [0, 1])
        self.assertEqual(dataset.cat_ids, [1, 2])

    def test_obj365v1_annotation_ids_unique(self):
        # test annotation ids not unique error
        metainfo = dict(classes=('car', ), task_name='new_task')
        with self.assertRaisesRegex(AssertionError, 'are not unique!'):
            Objects365V2Dataset(
                data_prefix=dict(img='imgs'),
                ann_file='tests/data/coco_wrong_format_sample.json',
                metainfo=metainfo,
                pipeline=[])
