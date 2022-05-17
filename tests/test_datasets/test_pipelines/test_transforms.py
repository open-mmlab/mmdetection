# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
import unittest

import mmcv
import numpy as np

from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
from mmdet.core.mask import BitmapMasks
from mmdet.datasets.pipelines import (Expand, MinIoURandomCrop,
                                      PhotoMetricDistortion, RandomFlip,
                                      Resize)
from .utils import create_random_bboxes


class TestResize(unittest.TestCase):

    def setUp(self):
        """Setup the model and optimizer which are used in every test method.

        TestCase calls functions in this order: setUp() -> testMethod() ->
        tearDown() -> cleanUp()
        """
        rng = np.random.RandomState(0)
        self.data_info1 = dict(
            img=np.random.random((1333, 800, 3)),
            gt_seg_map=np.random.random((1333, 800, 3)),
            gt_bboxes=np.array([[0, 0, 112, 112]]),
            gt_masks=BitmapMasks(
                rng.rand(1, 1333, 800), height=1333, width=800))
        self.data_info2 = dict(
            img=np.random.random((300, 400, 3)),
            gt_bboxes=np.array([[200, 150, 600, 450]]))

    def test_resize(self):
        # test keep_ratio is True
        transform = Resize(scale=(2000, 2000), keep_ratio=True)
        results = transform(copy.deepcopy(self.data_info1))
        self.assertEqual(results['img_shape'], (2000, 1200))
        self.assertEqual(results['scale'], (1200, 2000))
        self.assertEqual(results['scale_factor'], (1200 / 800, 2000 / 1333))

        # test resize_bboxes/seg/masks
        transform = Resize(scale_factor=(1.5, 2))
        results = transform(copy.deepcopy(self.data_info1))
        self.assertTrue((results['gt_bboxes'] == np.array([[0, 0, 168,
                                                            224]])).all())
        self.assertEqual(results['gt_masks'].height, 2666)
        self.assertEqual(results['gt_masks'].width, 1200)
        self.assertEqual(results['gt_seg_map'].shape[:2], (2666, 1200))

        # test clip_object_border = False
        transform = Resize(scale=(200, 150), clip_object_border=False)
        results = transform(self.data_info2)
        self.assertTrue((results['gt_bboxes'] == np.array([100, 75, 300,
                                                           225])).all())

    def test_repr(self):
        transform = Resize(scale=(2000, 2000), keep_ratio=True)
        self.assertEqual(
            repr(transform), ('Resize(scale=(2000, 2000), '
                              'scale_factor=None, keep_ratio=True, '
                              'clip_object_border=True), backend=cv2), '
                              'interpolation=bilinear)'))


class TestRandomFlip(unittest.TestCase):

    def setUp(self):
        """Setup the model and optimizer which are used in every test method.

        TestCase calls functions in this order: setUp() -> testMethod() ->
        tearDown() -> cleanUp()
        """
        rng = np.random.RandomState(0)
        self.results = {
            'img': np.random.random((224, 224, 3)),
            'gt_bboxes': np.array([[0, 1, 100, 101]]),
            'gt_masks':
            BitmapMasks(rng.rand(1, 224, 224), height=224, width=224),
            'gt_seg_map': np.random.random((224, 224))
        }

    def test_transform(self):
        transform = RandomFlip(1.0)
        results_update = transform.transform(copy.deepcopy(self.results))
        self.assertTrue(
            (results_update['gt_bboxes'] == np.array([[124, 1, 224,
                                                       101]])).all())

    def test_repr(self):
        transform = RandomFlip(0.1)
        transform_str = str(transform)
        self.assertIsInstance(transform_str, str)


class TestMinIoURandomCrop(unittest.TestCase):

    def test_transform(self):
        results = dict()
        img = mmcv.imread(
            osp.join(osp.dirname(__file__), '../../data/color.jpg'), 'color')
        results['img'] = img
        results['img_shape'] = img.shape[:2]
        gt_bboxes = create_random_bboxes(1, results['img_shape'][1],
                                         results['img_shape'][0])
        results['gt_labels'] = np.ones(gt_bboxes.shape[0], dtype=np.int64)
        results['gt_bboxes'] = gt_bboxes
        transform = MinIoURandomCrop()
        results = transform.transform(copy.deepcopy(results))

        self.assertEqual(results['gt_labels'].shape[0],
                         results['gt_bboxes'].shape[0])
        self.assertEqual(results['gt_labels'].dtype, np.int64)
        self.assertEqual(results['gt_bboxes'].dtype, np.float32)

        patch = np.array(
            [0, 0, results['img_shape'][1], results['img_shape'][0]])
        ious = bbox_overlaps(patch.reshape(-1, 4),
                             results['gt_bboxes']).reshape(-1)
        mode = transform.mode
        if mode == 1:
            self.assertTrue(np.equal(results['gt_bboxes'], gt_bboxes).all())
        else:
            self.assertTrue((ious >= mode).all())

    def test_repr(self):
        transform = MinIoURandomCrop()
        self.assertEqual(
            repr(transform), ('MinIoURandomCrop'
                              '(min_ious=(0.1, 0.3, 0.5, 0.7, 0.9), '
                              'min_crop_size=0.3, '
                              'bbox_clip_border=True)'))


class TestPhotoMetricDistortion(unittest.TestCase):

    def test_transform(self):
        img = mmcv.imread(
            osp.join(osp.dirname(__file__), '../../data/color.jpg'), 'color')
        transform = PhotoMetricDistortion()

        # test uint8 input
        results = dict()
        results['img'] = img
        results = transform.transform(copy.deepcopy(results))
        self.assertEqual(results['img'].dtype, np.float32)

        # test float32 input
        results = dict()
        results['img'] = img.astype(np.float32)
        results = transform.transform(copy.deepcopy(results))
        self.assertEqual(results['img'].dtype, np.float32)

    def test_repr(self):
        transform = PhotoMetricDistortion()
        self.assertEqual(
            repr(transform), ('PhotoMetricDistortion'
                              '(brightness_delta=32, '
                              'contrast_range=(0.5, 1.5), '
                              'saturation_range=(0.5, 1.5), '
                              'hue_delta=18)'))


class TestExpand(unittest.TestCase):

    def setUp(self):
        """Setup the model and optimizer which are used in every test method.

        TestCase calls functions in this order: setUp() -> testMethod() ->
        tearDown() -> cleanUp()
        """
        rng = np.random.RandomState(0)
        self.results = {
            'img': np.random.random((224, 224, 3)),
            'img_shape': (224, 224),
            'gt_bboxes': np.array([[0, 1, 100, 101]]),
            'gt_masks':
            BitmapMasks(rng.rand(1, 224, 224), height=224, width=224),
            'gt_seg_map': np.random.random((224, 224))
        }

    def test_transform(self):

        transform = Expand()
        results = transform.transform(copy.deepcopy(self.results))
        self.assertEqual(
            results['img_shape'],
            (results['gt_masks'].height, results['gt_masks'].width))
        self.assertEqual(results['img_shape'], results['gt_seg_map'].shape)

    def test_repr(self):
        transform = Expand()
        self.assertEqual(
            repr(transform), ('Expand'
                              '(mean=(0, 0, 0), to_rgb=True, '
                              'ratio_range=(1, 4), '
                              'seg_ignore_label=None, '
                              'prob=0.5)'))
