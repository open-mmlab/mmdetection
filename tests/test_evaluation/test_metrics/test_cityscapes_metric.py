import os
import os.path as osp
import tempfile
import unittest

import numpy as np
import torch
from PIL import Image

from mmdet.evaluation import CityScapesMetric

try:
    import cityscapesscripts
except ImportError:
    cityscapesscripts = None


class TestCityScapesMetric(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp_dir.cleanup()

    @unittest.skipIf(cityscapesscripts is None,
                     'cityscapesscripts is not installed.')
    def test_init(self):
        # test with outfile_prefix = None
        with self.assertRaises(AssertionError):
            CityScapesMetric(outfile_prefix=None)

    @unittest.skipIf(cityscapesscripts is None,
                     'cityscapesscripts is not installed.')
    def test_evaluate(self):
        dummy_mask1 = np.zeros((1, 20, 20), dtype=np.uint8)
        dummy_mask1[:, :10, :10] = 1
        dummy_mask2 = np.zeros((1, 20, 20), dtype=np.uint8)
        dummy_mask2[:, :10, :10] = 1

        self.outfile_prefix = osp.join(self.tmp_dir.name, 'test')
        self.seg_prefix = osp.join(self.tmp_dir.name, 'cityscapes/gtFine/val')
        city = 'lindau'
        sequenceNb = '000000'
        frameNb = '000019'
        img_name1 = f'{city}_{sequenceNb}_{frameNb}_gtFine_instanceIds.png'
        img_path1 = osp.join(self.seg_prefix, city, img_name1)

        frameNb = '000020'
        img_name2 = f'{city}_{sequenceNb}_{frameNb}_gtFine_instanceIds.png'
        img_path2 = osp.join(self.seg_prefix, city, img_name2)
        os.makedirs(osp.join(self.seg_prefix, city))

        masks1 = np.zeros((20, 20), dtype=np.int32)
        masks1[:10, :10] = 24 * 1000
        Image.fromarray(masks1).save(img_path1)

        masks2 = np.zeros((20, 20), dtype=np.int32)
        masks2[:10, :10] = 24 * 1000 + 1
        Image.fromarray(masks2).save(img_path2)

        data_samples = [{
            'img_path': img_path1,
            'pred_instances': {
                'scores': torch.from_numpy(np.array([1.0])),
                'labels': torch.from_numpy(np.array([0])),
                'masks': torch.from_numpy(dummy_mask1)
            }
        }, {
            'img_path': img_path2,
            'pred_instances': {
                'scores': torch.from_numpy(np.array([0.98])),
                'labels': torch.from_numpy(np.array([1])),
                'masks': torch.from_numpy(dummy_mask2)
            }
        }]

        target = {'cityscapes/mAP': 0.5, 'cityscapes/AP@50': 0.5}
        metric = CityScapesMetric(
            seg_prefix=self.seg_prefix,
            format_only=False,
            outfile_prefix=self.outfile_prefix)
        metric.dataset_meta = dict(
            classes=('person', 'rider', 'car', 'truck', 'bus', 'train',
                     'motorcycle', 'bicycle'))
        metric.process({}, data_samples)
        results = metric.evaluate(size=2)
        self.assertDictEqual(results, target)
        del metric
        self.assertTrue(not osp.exists('{self.outfile_prefix}.results'))

        # test format_only
        metric = CityScapesMetric(
            seg_prefix=self.seg_prefix,
            format_only=True,
            outfile_prefix=self.outfile_prefix)
        metric.dataset_meta = dict(
            classes=('person', 'rider', 'car', 'truck', 'bus', 'train',
                     'motorcycle', 'bicycle'))
        metric.process({}, data_samples)
        results = metric.evaluate(size=2)
        self.assertDictEqual(results, dict())
