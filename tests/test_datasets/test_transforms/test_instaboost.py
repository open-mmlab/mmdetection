import os.path as osp
import unittest

import numpy as np

from mmdet.registry import TRANSFORMS
from mmdet.utils import register_all_modules

register_all_modules()


class TestInstaboost(unittest.TestCase):

    def setUp(self):
        """Setup the model and optimizer which are used in every test method.

        TestCase calls functions in this order: setUp() -> testMethod() ->
        tearDown() -> cleanUp()
        """
        img_path = osp.join(osp.dirname(__file__), '../../data/gray.jpg')
        self.results = {
            'img_path':
            img_path,
            'img_shape': (300, 400),
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

    def test_transform(self):
        load = TRANSFORMS.build(dict(type='LoadImageFromFile'))
        instaboost_transform = TRANSFORMS.build(dict(type='InstaBoost'))

        # Execute transforms
        results = load(self.results)
        results = instaboost_transform(results)

        self.assertEqual(results['img'].dtype, np.uint8)
        self.assertIn('instances', results)

    def test_repr(self):
        instaboost_transform = TRANSFORMS.build(dict(type='InstaBoost'))

        self.assertEqual(
            repr(instaboost_transform), 'InstaBoost(aug_ratio=0.5)')
