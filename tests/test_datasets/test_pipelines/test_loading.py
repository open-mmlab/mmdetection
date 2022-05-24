# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
import os.path as osp
import sys
import unittest
from unittest.mock import MagicMock, Mock, patch

import mmcv
import numpy as np

from mmdet.core.evaluation import INSTANCE_OFFSET
from mmdet.core.mask import BitmapMasks, PolygonMasks
from mmdet.datasets.pipelines import FilterAnnotations, LoadAnnotations


class TestLoadAnnotations(unittest.TestCase):

    def setUp(self):
        """Setup the model and optimizer which are used in every test method.

        TestCase calls functions in this order: setUp() -> testMethod() ->
        tearDown() -> cleanUp()
        """
        data_prefix = osp.join(osp.dirname(__file__), '../../data')
        seg_map = osp.join(data_prefix, 'gray.jpg')
        self.results = {
            'img_shape': (300, 400),
            'seg_map_path':
            seg_map,
            'instances': [{
                'bbox': [0, 0, 10, 20],
                'bbox_label': 1,
                'mask': [[0, 0, 0, 20, 10, 20, 10, 0]],
                'ignore_flag': 0
            }, {
                'bbox': [10, 10, 110, 120],
                'bbox_label': 2,
                'mask': [[10, 10, 110, 10, 110, 120, 110, 10]],
                'ignore_flag': 0
            }, {
                'bbox': [50, 50, 60, 80],
                'bbox_label': 2,
                'mask': [[50, 50, 60, 50, 60, 80, 50, 80]],
                'ignore_flag': 1
            }]
        }

    def test_load_bboxes(self):
        transform = LoadAnnotations(
            with_bbox=True,
            with_label=False,
            with_seg=False,
            with_mask=False,
        )
        results = transform(copy.deepcopy(self.results))
        self.assertIn('gt_bboxes', results)
        self.assertTrue((results['gt_bboxes'] == np.array([[0, 0, 10, 20],
                                                           [10, 10, 110, 120],
                                                           [50, 50, 60,
                                                            80]])).all())
        self.assertEqual(results['gt_bboxes'].dtype, np.float32)
        self.assertTrue((results['gt_ignore_flags'] == np.array([0, 0,
                                                                 1])).all())
        self.assertEqual(results['gt_ignore_flags'].dtype, np.bool)

    def test_load_denorm_bboxes(self):
        transform = LoadAnnotations(
            with_bbox=True,
            with_label=False,
            with_seg=False,
            with_mask=False,
            denorm_bbox=True)
        results = transform(copy.deepcopy(self.results))
        self.assertIn('gt_bboxes', results)
        self.assertTrue(
            (results['gt_bboxes'] == np.array([[0, 0, 4000, 6000],
                                               [4000, 3000, 44000, 36000],
                                               [20000, 15000, 24000,
                                                24000]])).all())
        self.assertEqual(results['gt_bboxes'].dtype, np.float32)

    def test_load_labels(self):
        transform = LoadAnnotations(
            with_bbox=False,
            with_label=True,
            with_seg=False,
            with_mask=False,
        )
        results = transform(copy.deepcopy(self.results))
        self.assertIn('gt_bboxes_labels', results)
        self.assertTrue((results['gt_bboxes_labels'] == np.array([1, 2,
                                                                  2])).all())
        self.assertEqual(results['gt_bboxes_labels'].dtype, np.int64)

    def test_load_mask(self):
        transform = LoadAnnotations(
            with_bbox=False,
            with_label=False,
            with_seg=False,
            with_mask=True,
            poly2mask=False)
        results = transform(copy.deepcopy(self.results))
        self.assertIn('gt_masks', results)
        self.assertEqual(len(results['gt_masks']), 3)
        self.assertIsInstance(results['gt_masks'], PolygonMasks)

    def test_load_mask_poly2mask(self):
        transform = LoadAnnotations(
            with_bbox=False,
            with_label=False,
            with_seg=False,
            with_mask=True,
            poly2mask=True)
        results = transform(copy.deepcopy(self.results))
        self.assertIn('gt_masks', results)
        self.assertEqual(len(results['gt_masks']), 3)
        self.assertIsInstance(results['gt_masks'], BitmapMasks)

    def test_repr(self):
        transform = LoadAnnotations(
            with_bbox=True,
            with_label=False,
            with_seg=False,
            with_mask=False,
        )
        self.assertEqual(
            repr(transform), ('LoadAnnotations(with_bbox=True, '
                              'with_label=False, with_mask=False, '
                              'with_seg=False, poly2mask=True, '
                              "imdecode_backend='cv2', "
                              "file_client_args={'backend': 'disk'})"))


class TestFilterAnnotations(unittest.TestCase):

    def setUp(self):
        """Setup the model and optimizer which are used in every test method.

        TestCase calls functions in this order: setUp() -> testMethod() ->
        tearDown() -> cleanUp()
        """
        rng = np.random.RandomState(0)
        self.results = {
            'img':
            np.random.random((224, 224, 3)),
            'img_shape': (224, 224),
            'gt_bboxes_labels':
            np.array([1, 2, 3], dtype=np.int64),
            'gt_bboxes':
            np.array([[10, 10, 20, 20], [20, 20, 40, 40], [40, 40, 80, 80]]),
            'gt_ignore_flags':
            np.array([0, 0, 1], dtype=np.bool8),
            'gt_masks':
            BitmapMasks(rng.rand(3, 224, 224), height=224, width=224),
        }

    def test_transform(self):
        # test keep_empty = True
        transform = FilterAnnotations(
            min_gt_bbox_wh=(50, 50),
            keep_empty=True,
        )
        results = transform(copy.deepcopy(self.results))
        self.assertIsNone(results)

        # test keep_empty = False
        transform = FilterAnnotations(
            min_gt_bbox_wh=(50, 50),
            keep_empty=False,
        )
        results = transform(copy.deepcopy(self.results))
        self.assertTrue(isinstance(results, dict))

        # test filter annotations
        transform = FilterAnnotations(min_gt_bbox_wh=(15, 15), )
        results = transform(copy.deepcopy(self.results))
        self.assertIsInstance(results, dict)
        self.assertTrue((results['gt_bboxes_labels'] == np.array([2,
                                                                  3])).all())
        self.assertTrue((results['gt_bboxes'] == np.array([[20, 20, 40, 40],
                                                           [40, 40, 80,
                                                            80]])).all())
        self.assertTrue(len(results['gt_masks']) == 2)
        self.assertTrue(len(results['gt_ignore_flags'] == 2))

    def test_repr(self):
        transform = FilterAnnotations(
            min_gt_bbox_wh=(1, 1),
            keep_empty=False,
        )
        self.assertEqual(
            repr(transform), ('FilterAnnotations(min_gt_bbox_wh=(1, 1), '
                              'keep_empty=False)'))


class TestLoadPanopticAnnotations(unittest.TestCase):

    def setUp(self):
        seg_map = np.zeros((10, 10), dtype=np.int32)
        seg_map[:5, :10] = 1 + 10 * INSTANCE_OFFSET
        seg_map[5:10, :5] = 4 + 11 * INSTANCE_OFFSET
        seg_map[5:10, 5:10] = 6 + 0 * INSTANCE_OFFSET
        rgb_seg_map = np.zeros((10, 10, 3), dtype=np.uint8)
        rgb_seg_map[:, :, 0] = seg_map / (256 * 256)
        rgb_seg_map[:, :, 1] = seg_map % (256 * 256) / 256
        rgb_seg_map[:, :, 2] = seg_map % 256
        self.seg_map_path = './1.png'
        mmcv.imwrite(rgb_seg_map, self.seg_map_path)

        self.seg_map = seg_map
        self.rgb_seg_map = rgb_seg_map
        self.results = {
            'img_shape': (10, 10),
            'instances': [{
                'bbox': [0, 0, 10, 5],
                'bbox_label': 0,
                'ignore_flag': 0,
            }, {
                'bbox': [0, 5, 5, 10],
                'bbox_label': 1,
                'ignore_flag': 1,
            }],
            'segments_info': [
                {
                    'id': 1 + 10 * INSTANCE_OFFSET,
                    'category': 0,
                    'is_thing': True,
                },
                {
                    'id': 4 + 11 * INSTANCE_OFFSET,
                    'category': 1,
                    'is_thing': True,
                },
                {
                    'id': 6 + 0 * INSTANCE_OFFSET,
                    'category': 2,
                    'is_thing': False,
                },
            ],
            'seg_map_path':
            self.seg_map_path
        }

        self.gt_mask = BitmapMasks([
            (seg_map == 1 + 10 * INSTANCE_OFFSET).astype(np.uint8),
            (seg_map == 4 + 11 * INSTANCE_OFFSET).astype(np.uint8),
        ], 10, 10)
        self.gt_bboxes = np.array([[0, 0, 10, 5], [0, 5, 5, 10]],
                                  dtype=np.float32)
        self.gt_bboxes_labels = np.array([0, 1], dtype=np.int64)
        self.gt_ignore_flags = np.array([0, 1], dtype=bool)
        self.gt_seg_map = np.zeros((10, 10), dtype=np.int32)
        self.gt_seg_map[:5, :10] = 0
        self.gt_seg_map[5:10, :5] = 1
        self.gt_seg_map[5:10, 5:10] = 2

    def tearDown(self):
        os.remove(self.seg_map_path)

    def test_init(self):
        from mmdet.datasets.pipelines import LoadPanopticAnnotations
        with self.assertRaises(ImportError):
            LoadPanopticAnnotations()

    def test_transform(self):
        sys.modules['panopticapi'] = MagicMock()
        sys.modules['panopticapi.utils'] = MagicMock()
        from mmdet.datasets.pipelines import LoadPanopticAnnotations
        mock_rgb2id = Mock(return_value=self.seg_map)
        with patch('panopticapi.utils.rgb2id', mock_rgb2id):
            # test with all False
            transform = LoadPanopticAnnotations(
                with_bbox=False,
                with_label=False,
                with_mask=False,
                with_seg=False)
            results = transform(copy.deepcopy(self.results))
            self.assertDictEqual(results, self.results)
            # test with with_mask=True
            transform = LoadPanopticAnnotations(
                with_bbox=False,
                with_label=False,
                with_mask=True,
                with_seg=False)
            results = transform(copy.deepcopy(self.results))
            self.assertTrue(
                (results['gt_masks'].masks == self.gt_mask.masks).all())

            # test with with_seg=True
            transform = LoadPanopticAnnotations(
                with_bbox=False,
                with_label=False,
                with_mask=False,
                with_seg=True)
            results = transform(copy.deepcopy(self.results))
            self.assertNotIn('gt_masks', results)
            self.assertTrue((results['gt_seg_map'] == self.gt_seg_map).all())

            # test with all True
            transform = LoadPanopticAnnotations(
                with_bbox=True, with_label=True, with_mask=True, with_seg=True)
            results = transform(copy.deepcopy(self.results))
            self.assertTrue(
                (results['gt_masks'].masks == self.gt_mask.masks).all())
            self.assertTrue((results['gt_bboxes'] == self.gt_bboxes).all())
            self.assertTrue(
                (results['gt_bboxes_labels'] == self.gt_bboxes_labels).all())
            self.assertTrue(
                (results['gt_ignore_flags'] == self.gt_ignore_flags).all())
            self.assertTrue((results['gt_seg_map'] == self.gt_seg_map).all())
