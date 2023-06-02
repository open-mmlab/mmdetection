# Copyright (c) OpenMMLab. All rights reserved.
import os
import unittest

from mmengine.fileio import dump

from mmdet.datasets import CityscapesDataset


class TestCityscapesDataset(unittest.TestCase):

    def setUp(self) -> None:
        image1 = {
            'file_name': 'munster/munster_000102_000019_leftImg8bit.png',
            'height': 1024,
            'width': 2048,
            'segm_file': 'munster/munster_000102_000019_gtFine_labelIds.png',
            'id': 0
        }
        image2 = {
            'file_name': 'munster/munster_000157_000019_leftImg8bit.png',
            'height': 1024,
            'width': 2048,
            'segm_file': 'munster/munster_000157_000019_gtFine_labelIds.png',
            'id': 1
        }
        image3 = {
            'file_name': 'munster/munster_000139_000019_leftImg8bit.png',
            'height': 1024,
            'width': 2048,
            'segm_file': 'munster/munster_000139_000019_gtFine_labelIds.png',
            'id': 2
        }
        image4 = {
            'file_name': 'munster/munster_000034_000019_leftImg8bit.png',
            'height': 31,
            'width': 15,
            'segm_file': 'munster/munster_000034_000019_gtFine_labelIds.png',
            'id': 3
        }

        images = [image1, image2, image3, image4]

        categories = [{
            'id': 24,
            'name': 'person'
        }, {
            'id': 25,
            'name': 'rider'
        }, {
            'id': 26,
            'name': 'car'
        }]

        annotations = [
            {
                'iscrowd': 0,
                'category_id': 24,
                'bbox': [379.0, 435.0, 52.0, 124.0],
                'area': 2595,
                'segmentation': {
                    'size': [1024, 2048],
                    'counts': 'xxx'
                },
                'image_id': 0,
                'id': 0
            },
            {
                'iscrowd': 0,
                'category_id': 25,
                'bbox': [379.0, 435.0, 52.0, 124.0],
                'area': -1,
                'segmentation': {
                    'size': [1024, 2048],
                    'counts': 'xxx'
                },
                'image_id': 0,
                'id': 1
            },
            {
                'iscrowd': 0,
                'category_id': 26,
                'bbox': [379.0, 435.0, -1, 124.0],
                'area': 2,
                'segmentation': {
                    'size': [1024, 2048],
                    'counts': 'xxx'
                },
                'image_id': 0,
                'id': 2
            },
            {
                'iscrowd': 0,
                'category_id': 24,
                'bbox': [379.0, 435.0, 52.0, -1],
                'area': 2,
                'segmentation': {
                    'size': [1024, 2048],
                    'counts': 'xxx'
                },
                'image_id': 0,
                'id': 3
            },
            {
                'iscrowd': 0,
                'category_id': 1,
                'bbox': [379.0, 435.0, 52.0, 124.0],
                'area': 2595,
                'segmentation': {
                    'size': [1024, 2048],
                    'counts': 'xxx'
                },
                'image_id': 0,
                'id': 4
            },
            {
                'iscrowd': 1,
                'category_id': 26,
                'bbox': [379.0, 435.0, 52.0, 124.0],
                'area': 2595,
                'segmentation': {
                    'size': [1024, 2048],
                    'counts': 'xxx'
                },
                'image_id': 1,
                'id': 5
            },
            {
                'iscrowd': 0,
                'category_id': 26,
                'bbox': [379.0, 435.0, 10, 2],
                'area': 2595,
                'segmentation': {
                    'size': [1024, 2048],
                    'counts': 'xxx'
                },
                'image_id': 3,
                'id': 6
            },
        ]
        fake_json = {
            'images': images,
            'annotations': annotations,
            'categories': categories
        }
        self.json_name = 'cityscapes.json'
        dump(fake_json, self.json_name)

        self.metainfo = dict(classes=('person', 'rider', 'car'))

    def tearDown(self):
        os.remove(self.json_name)

    def test_cityscapes_dataset(self):
        dataset = CityscapesDataset(
            ann_file=self.json_name,
            data_prefix=dict(img='imgs'),
            metainfo=self.metainfo,
            filter_cfg=dict(filter_empty_gt=True, min_size=32),
            pipeline=[])
        self.assertEqual(dataset.metainfo['classes'], self.metainfo['classes'])
        dataset.full_init()
        # filter images of small size and images
        # with all illegal annotations
        self.assertEqual(len(dataset), 1)
        self.assertEqual(len(dataset.load_data_list()), 4)

        dataset = CityscapesDataset(
            ann_file=self.json_name,
            data_prefix=dict(img='imgs'),
            metainfo=self.metainfo,
            test_mode=True,
            filter_cfg=dict(filter_empty_gt=True, min_size=32),
            pipeline=[])
        dataset.full_init()
        # filter images of small size and images
        # with all illegal annotations
        self.assertEqual(len(dataset), 4)
        self.assertEqual(len(dataset.load_data_list()), 4)

    def test_cityscapes_dataset_without_filter_cfg(self):
        dataset = CityscapesDataset(
            ann_file=self.json_name,
            data_prefix=dict(img='imgs'),
            metainfo=self.metainfo,
            filter_cfg=None,
            pipeline=[])
        self.assertEqual(dataset.metainfo['classes'], self.metainfo['classes'])
        dataset.full_init()
        # filter images of small size and images
        # with all illegal annotations
        self.assertEqual(len(dataset), 4)
        self.assertEqual(len(dataset.load_data_list()), 4)

        dataset = CityscapesDataset(
            ann_file=self.json_name,
            data_prefix=dict(img='imgs'),
            metainfo=self.metainfo,
            test_mode=True,
            filter_cfg=None,
            pipeline=[])
        dataset.full_init()
        # filter images of small size and images
        # with all illegal annotations
        self.assertEqual(len(dataset), 4)
        self.assertEqual(len(dataset.load_data_list()), 4)
