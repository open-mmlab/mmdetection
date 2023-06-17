# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
import os.path as osp
import sys
import unittest
from unittest.mock import MagicMock, Mock, patch

import mmcv
import numpy as np

from mmdet.datasets.transforms import (FilterAnnotations, LoadAnnotations,
                                       LoadEmptyAnnotations,
                                       LoadImageFromNDArray,
                                       LoadMultiChannelImageFromFiles,
                                       LoadProposals, LoadTrackAnnotations)
from mmdet.evaluation import INSTANCE_OFFSET
from mmdet.structures.mask import BitmapMasks, PolygonMasks

try:
    import panopticapi
except ImportError:
    panopticapi = None


class TestLoadAnnotations(unittest.TestCase):

    def setUp(self):
        """Setup the model and optimizer which are used in every test method.

        TestCase calls functions in this order: setUp() -> testMethod() ->
        tearDown() -> cleanUp()
        """
        data_prefix = osp.join(osp.dirname(__file__), '../../data')
        seg_map = osp.join(data_prefix, 'gray.jpg')
        self.results = {
            'ori_shape': (300, 400),
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
            box_type=None)
        results = transform(copy.deepcopy(self.results))
        self.assertIn('gt_bboxes', results)
        self.assertTrue((results['gt_bboxes'] == np.array([[0, 0, 10, 20],
                                                           [10, 10, 110, 120],
                                                           [50, 50, 60,
                                                            80]])).all())
        self.assertEqual(results['gt_bboxes'].dtype, np.float32)
        self.assertTrue((results['gt_ignore_flags'] == np.array([0, 0,
                                                                 1])).all())
        self.assertEqual(results['gt_ignore_flags'].dtype, bool)

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
                              'backend_args=None)'))


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
        self.assertEqual(len(results['gt_masks']), 2)
        self.assertEqual(len(results['gt_ignore_flags']), 2)

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
            'ori_shape': (10, 10),
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

    @unittest.skipIf(panopticapi is not None, 'panopticapi is installed')
    def test_init_without_panopticapi(self):
        # test if panopticapi is not installed
        from mmdet.datasets.transforms import LoadPanopticAnnotations
        with self.assertRaisesRegex(
                ImportError,
                'panopticapi is not installed, please install it by'):
            LoadPanopticAnnotations()

    def test_transform(self):
        sys.modules['panopticapi'] = MagicMock()
        sys.modules['panopticapi.utils'] = MagicMock()
        from mmdet.datasets.transforms import LoadPanopticAnnotations
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
                with_bbox=True,
                with_label=True,
                with_mask=True,
                with_seg=True,
                box_type=None)
            results = transform(copy.deepcopy(self.results))
            self.assertTrue(
                (results['gt_masks'].masks == self.gt_mask.masks).all())
            self.assertTrue((results['gt_bboxes'] == self.gt_bboxes).all())
            self.assertTrue(
                (results['gt_bboxes_labels'] == self.gt_bboxes_labels).all())
            self.assertTrue(
                (results['gt_ignore_flags'] == self.gt_ignore_flags).all())
            self.assertTrue((results['gt_seg_map'] == self.gt_seg_map).all())


class TestLoadImageFromNDArray(unittest.TestCase):

    def setUp(self):
        """Setup the model and optimizer which are used in every test method.

        TestCase calls functions in this order: setUp() -> testMethod() ->
        tearDown() -> cleanUp()
        """
        self.results = {'img': np.zeros((256, 256, 3), dtype=np.uint8)}

    def test_transform(self):
        transform = LoadImageFromNDArray()
        results = transform(copy.deepcopy(self.results))
        self.assertEqual(results['img'].shape, (256, 256, 3))
        self.assertEqual(results['img'].dtype, np.uint8)
        self.assertEqual(results['img_shape'], (256, 256))
        self.assertEqual(results['ori_shape'], (256, 256))

        # to_float32
        transform = LoadImageFromNDArray(to_float32=True)
        results = transform(copy.deepcopy(results))
        self.assertEqual(results['img'].dtype, np.float32)

    def test_repr(self):
        transform = LoadImageFromNDArray()
        self.assertEqual(
            repr(transform), ('LoadImageFromNDArray('
                              'ignore_empty=False, '
                              'to_float32=False, '
                              "color_type='color', "
                              "imdecode_backend='cv2', "
                              'backend_args=None)'))


class TestLoadMultiChannelImageFromFiles(unittest.TestCase):

    def setUp(self):
        """Setup the model and optimizer which are used in every test method.

        TestCase calls functions in this order: setUp() -> testMethod() ->
        tearDown() -> cleanUp()
        """
        self.img_path = []
        for i in range(4):
            img_channel_path = f'./part_{i}.jpg'
            img_channel = np.zeros((10, 10), dtype=np.uint8)
            mmcv.imwrite(img_channel, img_channel_path)
            self.img_path.append(img_channel_path)
        self.results = {'img_path': self.img_path}

    def tearDown(self):
        for filename in self.img_path:
            os.remove(filename)

    def test_transform(self):
        transform = LoadMultiChannelImageFromFiles()
        results = transform(copy.deepcopy(self.results))
        self.assertEqual(results['img'].shape, (10, 10, 4))
        self.assertEqual(results['img'].dtype, np.uint8)
        self.assertEqual(results['img_shape'], (10, 10))
        self.assertEqual(results['ori_shape'], (10, 10))

        # to_float32
        transform = LoadMultiChannelImageFromFiles(to_float32=True)
        results = transform(copy.deepcopy(results))
        self.assertEqual(results['img'].dtype, np.float32)

    def test_rper(self):
        transform = LoadMultiChannelImageFromFiles()
        self.assertEqual(
            repr(transform), ('LoadMultiChannelImageFromFiles('
                              'to_float32=False, '
                              "color_type='unchanged', "
                              "imdecode_backend='cv2', "
                              'backend_args=None)'))


class TestLoadProposals(unittest.TestCase):

    def test_transform(self):
        transform = LoadProposals()
        results = {
            'proposals':
            dict(
                bboxes=np.zeros((5, 4), dtype=np.int64),
                scores=np.zeros((5, ), dtype=np.int64))
        }
        results = transform(results)
        self.assertEqual(results['proposals'].dtype, np.float32)
        self.assertEqual(results['proposals'].shape[-1], 4)
        self.assertEqual(results['proposals_scores'].dtype, np.float32)

        #  bboxes.shape[1] should be 4
        results = {'proposals': dict(bboxes=np.zeros((5, 5), dtype=np.int64))}
        with self.assertRaises(AssertionError):
            transform(results)

        # bboxes.shape[0] should equal to scores.shape[0]
        results = {
            'proposals':
            dict(
                bboxes=np.zeros((5, 4), dtype=np.int64),
                scores=np.zeros((3, ), dtype=np.int64))
        }
        with self.assertRaises(AssertionError):
            transform(results)

        # empty bboxes
        results = {
            'proposals': dict(bboxes=np.zeros((0, 4), dtype=np.float32))
        }
        results = transform(results)
        excepted_proposals = np.zeros((0, 4), dtype=np.float32)
        excepted_proposals_scores = np.zeros(0, dtype=np.float32)
        self.assertTrue((results['proposals'] == excepted_proposals).all())
        self.assertTrue(
            (results['proposals_scores'] == excepted_proposals_scores).all())

        transform = LoadProposals(num_max_proposals=2)
        results = {
            'proposals':
            dict(
                bboxes=np.zeros((5, 4), dtype=np.int64),
                scores=np.zeros((5, ), dtype=np.int64))
        }
        results = transform(results)
        self.assertEqual(results['proposals'].shape[0], 2)

    def test_repr(self):
        transform = LoadProposals()
        self.assertEqual(
            repr(transform), 'LoadProposals(num_max_proposals=None)')


class TestLoadEmptyAnnotations(unittest.TestCase):

    def test_transform(self):
        transform = LoadEmptyAnnotations(
            with_bbox=True, with_label=True, with_mask=True, with_seg=True)
        results = {'img_shape': (224, 224)}
        results = transform(results)
        self.assertEqual(results['gt_bboxes'].dtype, np.float32)
        self.assertEqual(results['gt_bboxes'].shape[-1], 4)
        self.assertEqual(results['gt_ignore_flags'].dtype, bool)
        self.assertEqual(results['gt_bboxes_labels'].dtype, np.int64)
        self.assertEqual(results['gt_masks'].masks.dtype, np.uint8)
        self.assertEqual(results['gt_masks'].masks.shape[-2:],
                         results['img_shape'])
        self.assertEqual(results['gt_seg_map'].dtype, np.uint8)
        self.assertEqual(results['gt_seg_map'].shape, results['img_shape'])

    def test_repr(self):
        transform = LoadEmptyAnnotations()
        self.assertEqual(
            repr(transform), 'LoadEmptyAnnotations(with_bbox=True, '
            'with_label=True, '
            'with_mask=False, '
            'with_seg=False, '
            'seg_ignore_label=255)')


class TestLoadTrackAnnotations(unittest.TestCase):

    def setUp(self):
        data_prefix = osp.join(osp.dirname(__file__), '../data')
        seg_map = osp.join(data_prefix, 'grayscale.jpg')
        self.results = {
            'seg_map_path':
            seg_map,
            'instances': [{
                'bbox': [0, 0, 10, 20],
                'bbox_label': 1,
                'instance_id': 100,
                'keypoints': [1, 2, 3]
            }, {
                'bbox': [10, 10, 110, 120],
                'bbox_label': 2,
                'instance_id': 102,
                'keypoints': [4, 5, 6]
            }]
        }

    def test_load_instances_id(self):
        transform = LoadTrackAnnotations(
            with_bbox=False,
            with_label=True,
            with_seg=False,
            with_keypoints=False,
        )
        results = transform(copy.deepcopy(self.results))
        assert 'gt_instances_ids' in results
        assert (results['gt_instances_ids'] == np.array([100, 102])).all()
        assert results['gt_instances_ids'].dtype == np.int32

    def test_repr(self):
        transform = LoadTrackAnnotations(
            with_bbox=True, with_label=False, with_seg=False, with_mask=False)
        assert repr(transform) == ('LoadTrackAnnotations(with_bbox=True, '
                                   'with_label=False, with_mask=False,'
                                   ' with_seg=False, poly2mask=True,'
                                   " imdecode_backend='cv2', "
                                   'file_client_args=None)')
