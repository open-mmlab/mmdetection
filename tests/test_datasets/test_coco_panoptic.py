# Copyright (c) OpenMMLab. All rights reserved.
import os
import unittest

from mmengine.fileio import dump

from mmdet.datasets import CocoPanopticDataset


class TestCocoPanopticDataset(unittest.TestCase):

    def setUp(self):
        image1 = {
            'id': 0,
            'width': 640,
            'height': 640,
            'file_name': 'fake_name1.jpg',
        }

        image2 = {
            'id': 1,
            'width': 640,
            'height': 800,
            'file_name': 'fake_name2.jpg',
        }

        image3 = {
            'id': 2,
            'width': 31,
            'height': 40,
            'file_name': 'fake_name3.jpg',
        }

        image4 = {
            'id': 3,
            'width': 400,
            'height': 400,
            'file_name': 'fake_name4.jpg',
        }
        images = [image1, image2, image3, image4]

        annotations = [
            {
                'segments_info': [{
                    'id': 1,
                    'category_id': 0,
                    'area': 400,
                    'bbox': [50, 60, 20, 20],
                    'iscrowd': 0
                }, {
                    'id': 2,
                    'category_id': 1,
                    'area': 900,
                    'bbox': [100, 120, 30, 30],
                    'iscrowd': 0
                }, {
                    'id': 3,
                    'category_id': 2,
                    'iscrowd': 0,
                    'bbox': [1, 189, 612, 285],
                    'area': 70036
                }],
                'file_name':
                'fake_name1.jpg',
                'image_id':
                0
            },
            {
                'segments_info': [
                    {
                        # Different to instance style json, there
                        # are duplicate ids in panoptic style json
                        'id': 1,
                        'category_id': 0,
                        'area': 400,
                        'bbox': [50, 60, 20, 20],
                        'iscrowd': 0
                    },
                    {
                        'id': 4,
                        'category_id': 1,
                        'area': 900,
                        'bbox': [100, 120, 30, 30],
                        'iscrowd': 1
                    },
                    {
                        'id': 5,
                        'category_id': 2,
                        'iscrowd': 0,
                        'bbox': [100, 200, 200, 300],
                        'area': 66666
                    },
                    {
                        'id': 6,
                        'category_id': 0,
                        'iscrowd': 0,
                        'bbox': [1, 189, -10, 285],
                        'area': -2
                    },
                    {
                        'id': 10,
                        'category_id': 0,
                        'iscrowd': 0,
                        'bbox': [1, 189, 10, -285],
                        'area': 100
                    }
                ],
                'file_name':
                'fake_name2.jpg',
                'image_id':
                1
            },
            {
                'segments_info': [{
                    'id': 7,
                    'category_id': 0,
                    'area': 25,
                    'bbox': [0, 0, 5, 5],
                    'iscrowd': 0
                }],
                'file_name':
                'fake_name3.jpg',
                'image_id':
                2
            },
            {
                'segments_info': [{
                    'id': 8,
                    'category_id': 0,
                    'area': 25,
                    'bbox': [0, 0, 400, 400],
                    'iscrowd': 1
                }],
                'file_name':
                'fake_name4.jpg',
                'image_id':
                3
            }
        ]

        categories = [{
            'id': 0,
            'name': 'car',
            'supercategory': 'car',
            'isthing': 1
        }, {
            'id': 1,
            'name': 'person',
            'supercategory': 'person',
            'isthing': 1
        }, {
            'id': 2,
            'name': 'wall',
            'supercategory': 'wall',
            'isthing': 0
        }]

        fake_json = {
            'images': images,
            'annotations': annotations,
            'categories': categories
        }
        self.json_name = 'coco_panoptic.json'
        dump(fake_json, self.json_name)

        self.metainfo = dict(
            classes=('person', 'car', 'wall'),
            thing_classes=('person', 'car'),
            stuff_classes=('wall', ))

    def tearDown(self):
        os.remove(self.json_name)

    def test_coco_panoptic_dataset(self):
        dataset = CocoPanopticDataset(
            data_root='./',
            ann_file=self.json_name,
            data_prefix=dict(img='imgs', seg='seg'),
            metainfo=self.metainfo,
            filter_cfg=dict(filter_empty_gt=True, min_size=32),
            pipeline=[])
        self.assertEqual(dataset.metainfo['classes'], self.metainfo['classes'])
        self.assertEqual(dataset.metainfo['thing_classes'],
                         self.metainfo['thing_classes'])
        self.assertEqual(dataset.metainfo['stuff_classes'],
                         self.metainfo['stuff_classes'])
        dataset.full_init()
        # filter images of small size and images
        # with all illegal annotations
        self.assertEqual(len(dataset), 2)
        self.assertEqual(len(dataset.load_data_list()), 4)
        # test mode
        dataset = CocoPanopticDataset(
            data_root='./',
            ann_file=self.json_name,
            data_prefix=dict(img='imgs', seg='seg'),
            metainfo=self.metainfo,
            test_mode=True,
            filter_cfg=dict(filter_empty_gt=True, min_size=32),
            pipeline=[])
        self.assertEqual(dataset.metainfo['classes'], self.metainfo['classes'])
        self.assertEqual(dataset.metainfo['thing_classes'],
                         self.metainfo['thing_classes'])
        self.assertEqual(dataset.metainfo['stuff_classes'],
                         self.metainfo['stuff_classes'])
        dataset.full_init()
        # filter images of small size and images
        # with all illegal annotations
        self.assertEqual(len(dataset), 4)
        self.assertEqual(len(dataset.load_data_list()), 4)

    def test_coco_panoptic_dataset_without_filter_cfg(self):
        dataset = CocoPanopticDataset(
            data_root='./',
            ann_file=self.json_name,
            data_prefix=dict(img='imgs', seg='seg'),
            metainfo=self.metainfo,
            filter_cfg=None,
            pipeline=[])
        self.assertEqual(dataset.metainfo['classes'], self.metainfo['classes'])
        self.assertEqual(dataset.metainfo['thing_classes'],
                         self.metainfo['thing_classes'])
        self.assertEqual(dataset.metainfo['stuff_classes'],
                         self.metainfo['stuff_classes'])
        dataset.full_init()
        # filter images of small size and images
        # with all illegal annotations
        self.assertEqual(len(dataset), 4)
        self.assertEqual(len(dataset.load_data_list()), 4)

        # test mode
        dataset = CocoPanopticDataset(
            data_root='./',
            ann_file=self.json_name,
            data_prefix=dict(img='imgs', seg='seg'),
            metainfo=self.metainfo,
            filter_cfg=None,
            test_mode=True,
            pipeline=[])
        self.assertEqual(dataset.metainfo['classes'], self.metainfo['classes'])
        self.assertEqual(dataset.metainfo['thing_classes'],
                         self.metainfo['thing_classes'])
        self.assertEqual(dataset.metainfo['stuff_classes'],
                         self.metainfo['stuff_classes'])
        dataset.full_init()
        # filter images of small size and images
        # with all illegal annotations
        self.assertEqual(len(dataset), 4)
        self.assertEqual(len(dataset.load_data_list()), 4)
