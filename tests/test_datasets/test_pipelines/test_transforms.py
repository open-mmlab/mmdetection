# Copyright (c) OpenMMLab. All rights reserved.
import copy
import unittest

import numpy as np

from mmdet.core.mask import BitmapMasks
from mmdet.datasets.pipelines import RandomFlip, Resize


class TestResize(unittest.TestCase):

    def setUp(self):
        """Setup the model and optimizer which are used in every test method.

        TestCase calls functions in this order: setUp() -> testMethod() ->
        tearDown() -> cleanUp()
        """
        rng = np.random.RandomState(0)
        self.data_info1 = dict(
            height=2666,
            width=1200,
            img=np.random.random((1333, 800, 3)),
            gt_seg_map=np.random.random((1333, 800, 3)),
            gt_bboxes=np.array([[0, 0, 112, 112]]),
            gt_masks=BitmapMasks(
                rng.rand(3, 1333, 800), height=1333, width=800))
        self.data_info2 = dict(
            img=np.random.random((300, 400, 3)),
            gt_bboxes=np.array([[200, 150, 600, 450]]))

    def test_resize(self):
        # test keep_ratio is True
        transform = Resize(scale=(2000, 2000), keep_ratio=True)
        results = transform(copy.deepcopy(self.data_info1))
        self.assertEqual(results['img'].shape[:2], (2000, 1200))
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
            BitmapMasks(rng.rand(3, 224, 224), height=224, width=224),
            'gt_seg_map': np.random.random((224, 224, 3))
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
