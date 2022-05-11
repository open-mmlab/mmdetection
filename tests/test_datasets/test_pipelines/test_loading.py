# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
import unittest

import numpy as np

from mmdet.core.mask import BitmapMasks, PolygonMasks
from mmdet.datasets.pipelines import LoadAnnotations


class TestLoadAnnotations(unittest.TestCase):

    def setUp(self):
        """Setup the model and optimizer which are used in every test method.

        TestCase calls functions in this order: setUp() -> testMethod() ->
        tearDown() -> cleanUp()
        """
        data_prefix = osp.join(osp.dirname(__file__), '../../data')
        seg_map = osp.join(data_prefix, 'gray.jpg')
        self.results = {
            'height':
            300,
            'width':
            400,
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
