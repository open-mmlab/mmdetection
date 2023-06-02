# Copyright (c) OpenMMLab. All rights reserved.
import os
import unittest

from mmengine.fileio import dump

from mmdet.datasets import LVISV1Dataset, LVISV05Dataset

try:
    import lvis
except ImportError:
    lvis = None


class TestLVISDataset(unittest.TestCase):

    def setUp(self) -> None:

        image1 = {
            # ``coco_url`` for v1 only.
            'coco_url': 'http://images.cocodataset.org/train2017/0.jpg',
            # ``file_name`` for v0.5 only.
            'file_name': '0.jpg',
            'height': 1024,
            'width': 2048,
            'neg_category_ids': [],
            'not_exhaustive_category_ids': [],
            'id': 0
        }
        image2 = {
            'coco_url': 'http://images.cocodataset.org/train2017/1.jpg',
            'file_name': '1.jpg',
            'height': 1024,
            'width': 2048,
            'neg_category_ids': [],
            'not_exhaustive_category_ids': [],
            'id': 1
        }
        image3 = {
            'coco_url': 'http://images.cocodataset.org/train2017/2.jpg',
            'file_name': '2.jpg',
            'height': 1024,
            'width': 2048,
            'neg_category_ids': [],
            'not_exhaustive_category_ids': [],
            'id': 2
        }
        image4 = {
            'coco_url': 'http://images.cocodataset.org/train2017/3.jpg',
            'file_name': '3.jpg',
            'height': 31,
            'width': 15,
            'neg_category_ids': [],
            'not_exhaustive_category_ids': [],
            'id': 3
        }

        images = [image1, image2, image3, image4]

        categories = [{
            'id': 1,
            'name': 'aerosol_can',
            'frequency': 'c',
            'image_count': 64
        }, {
            'id': 2,
            'name': 'air_conditioner',
            'frequency': 'f',
            'image_count': 364
        }, {
            'id': 3,
            'name': 'airplane',
            'frequency': 'f',
            'image_count': 1911
        }]

        annotations = [
            {
                'category_id': 1,
                'bbox': [379.0, 435.0, 52.0, 124.0],
                'area': 2595,
                'segmentation': [[0.0, 0.0]],
                'image_id': 0,
                'id': 0
            },
            {
                'category_id': 2,
                'bbox': [379.0, 435.0, 52.0, 124.0],
                'area': -1,
                'segmentation': [[0.0, 0.0]],
                'image_id': 0,
                'id': 1
            },
            {
                'category_id': 3,
                'bbox': [379.0, 435.0, -1, 124.0],
                'area': 2,
                'segmentation': [[0.0, 0.0]],
                'image_id': 0,
                'id': 2
            },
            {
                'category_id': 1,
                'bbox': [379.0, 435.0, 52.0, -1],
                'area': 2,
                'segmentation': [[0.0, 0.0]],
                'image_id': 0,
                'id': 3
            },
            {
                'category_id': 1,
                'bbox': [379.0, 435.0, 52.0, 124.0],
                'area': 2595,
                'segmentation': [[0.0, 0.0]],
                'image_id': 0,
                'id': 4
            },
            {
                'category_id': 3,
                'bbox': [379.0, 435.0, 52.0, 124.0],
                'area': 2595,
                'segmentation': [[0.0, 0.0]],
                'image_id': 1,
                'id': 5
            },
            {
                'category_id': 3,
                'bbox': [379.0, 435.0, 10, 2],
                'area': 2595,
                'segmentation': [[0.0, 0.0]],
                'image_id': 3,
                'id': 6
            },
        ]
        fake_json = {
            'images': images,
            'annotations': annotations,
            'categories': categories
        }
        self.json_name = 'lvis.json'
        dump(fake_json, self.json_name)

        self.metainfo = dict(
            classes=('aerosol_can', 'air_conditioner', 'airplane'))

    def tearDown(self):
        os.remove(self.json_name)

    @unittest.skipIf(lvis is None, 'lvis is not installed.')
    def test_lvis05_dataset(self):
        dataset = LVISV05Dataset(
            ann_file=self.json_name,
            data_prefix=dict(img='imgs'),
            metainfo=self.metainfo,
            filter_cfg=dict(filter_empty_gt=True, min_size=32),
            pipeline=[])
        self.assertEqual(dataset.metainfo['classes'], self.metainfo['classes'])
        dataset.full_init()
        # filter images of small size and images
        # with all illegal annotations
        self.assertEqual(len(dataset), 2)
        self.assertEqual(len(dataset.load_data_list()), 4)

        dataset = LVISV05Dataset(
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

    @unittest.skipIf(lvis is None, 'lvis is not installed.')
    def test_lvis1_dataset(self):
        dataset = LVISV1Dataset(
            ann_file=self.json_name,
            data_prefix=dict(img='imgs'),
            metainfo=self.metainfo,
            filter_cfg=dict(filter_empty_gt=True, min_size=32),
            pipeline=[])
        self.assertEqual(dataset.metainfo['classes'], self.metainfo['classes'])
        dataset.full_init()
        # filter images of small size and images
        # with all illegal annotations
        self.assertEqual(len(dataset), 2)
        self.assertEqual(len(dataset.load_data_list()), 4)

        dataset = LVISV1Dataset(
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

    @unittest.skipIf(lvis is None, 'lvis is not installed.')
    def test_lvis1_dataset_without_filter_cfg(self):
        dataset = LVISV1Dataset(
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

        dataset = LVISV1Dataset(
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
