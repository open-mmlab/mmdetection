# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
import unittest

import numpy as np
from mmengine.data import InstanceData, PixelData

from mmdet.datasets.transforms import PackDetInputs
from mmdet.structures import DetDataSample
from mmdet.structures.mask import BitmapMasks


class TestPackDetInputs(unittest.TestCase):

    def setUp(self):
        """Setup the model and optimizer which are used in every test method.

        TestCase calls functions in this order: setUp() -> testMethod() ->
        tearDown() -> cleanUp()
        """
        data_prefix = osp.join(osp.dirname(__file__), '../../data')
        img_path = osp.join(data_prefix, 'color.jpg')
        rng = np.random.RandomState(0)
        self.results1 = {
            'img_id': 1,
            'img_path': img_path,
            'ori_shape': (300, 400),
            'img_shape': (600, 800),
            'scale_factor': 2.0,
            'flip': False,
            'img': rng.rand(300, 400),
            'gt_seg_map': rng.rand(300, 400),
            'gt_masks':
            BitmapMasks(rng.rand(3, 300, 400), height=300, width=400),
            'gt_bboxes_labels': rng.rand(3, ),
            'gt_ignore_flags': np.array([0, 0, 1], dtype=np.bool),
            'proposals': rng.rand(2, 4)
        }
        self.results2 = {
            'img_id': 1,
            'img_path': img_path,
            'ori_shape': (300, 400),
            'img_shape': (600, 800),
            'scale_factor': 2.0,
            'flip': False,
            'img': rng.rand(300, 400),
            'gt_seg_map': rng.rand(300, 400),
            'gt_masks':
            BitmapMasks(rng.rand(3, 300, 400), height=300, width=400),
            'gt_bboxes_labels': rng.rand(3, ),
            'proposals': rng.rand(2, 4)
        }
        self.meta_keys = ('img_id', 'img_path', 'ori_shape', 'scale_factor',
                          'flip')

    def test_transform(self):
        transform = PackDetInputs(meta_keys=self.meta_keys)
        results = transform(copy.deepcopy(self.results1))
        self.assertIn('data_sample', results)
        self.assertIsInstance(results['data_sample'], DetDataSample)
        self.assertIsInstance(results['data_sample'].gt_instances,
                              InstanceData)
        self.assertIsInstance(results['data_sample'].ignored_instances,
                              InstanceData)
        self.assertEqual(len(results['data_sample'].gt_instances), 2)
        self.assertEqual(len(results['data_sample'].ignored_instances), 1)
        self.assertIsInstance(results['data_sample'].gt_sem_seg, PixelData)
        self.assertIsInstance(results['data_sample'].proposals, InstanceData)
        self.assertEqual(len(results['data_sample'].proposals), 2)
        self.assertIsInstance(results['data_sample'].proposals.bboxes,
                              np.ndarray)

    def test_transform_without_ignore(self):
        transform = PackDetInputs(meta_keys=self.meta_keys)
        results = transform(copy.deepcopy(self.results2))
        self.assertIn('data_sample', results)
        self.assertIsInstance(results['data_sample'], DetDataSample)
        self.assertIsInstance(results['data_sample'].gt_instances,
                              InstanceData)
        self.assertIsInstance(results['data_sample'].ignored_instances,
                              InstanceData)
        self.assertEqual(len(results['data_sample'].gt_instances), 3)
        self.assertEqual(len(results['data_sample'].ignored_instances), 0)
        self.assertIsInstance(results['data_sample'].gt_sem_seg, PixelData)
        self.assertIsInstance(results['data_sample'].proposals, InstanceData)
        self.assertEqual(len(results['data_sample'].proposals), 2)
        self.assertIsInstance(results['data_sample'].proposals.bboxes,
                              np.ndarray)

    def test_repr(self):
        transform = PackDetInputs(meta_keys=self.meta_keys)
        self.assertEqual(
            repr(transform), f'PackDetInputs(meta_keys={self.meta_keys})')
