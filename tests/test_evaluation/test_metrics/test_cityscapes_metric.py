import math
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

        # test with seg_prefix = None
        with self.assertRaises(AssertionError):
            CityScapesMetric(
                outfile_prefix='tmp/cityscapes/results', seg_prefix=None)

        # test with format_only=True, keep_results=False
        with self.assertRaises(AssertionError):
            CityScapesMetric(
                outfile_prefix='tmp/cityscapes/results',
                format_only=True,
                keep_results=False)

    @unittest.skipIf(cityscapesscripts is None,
                     'cityscapesscripts is not installed.')
    def test_evaluate(self):
        tmp_dir = tempfile.TemporaryDirectory()

        dataset_metas = {
            'classes': ('person', 'rider', 'car', 'truck', 'bus', 'train',
                        'motorcycle', 'bicycle')
        }

        # create dummy data
        self.seg_prefix = osp.join(tmp_dir.name, 'cityscapes', 'gtFine', 'val')
        os.makedirs(self.seg_prefix, exist_ok=True)
        data_samples = self._gen_fake_datasamples()

        # test single evaluation
        metric = CityScapesMetric(
            dataset_meta=dataset_metas,
            outfile_prefix=osp.join(tmp_dir.name, 'test'),
            seg_prefix=self.seg_prefix,
            keep_results=False,
            keep_gt_json=False,
            classwise=False)

        metric.process({}, data_samples)
        results = metric.evaluate()
        targets = {'cityscapes/mAP(%)': 50.0, 'cityscapes/AP50(%)': 50.0}
        self.assertDictEqual(results, targets)

        # test classwise result evaluation
        metric = CityScapesMetric(
            dataset_meta=dataset_metas,
            outfile_prefix=osp.join(tmp_dir.name, 'test'),
            seg_prefix=self.seg_prefix,
            keep_results=False,
            keep_gt_json=False,
            classwise=True)

        metric.process({}, data_samples)
        results = metric.evaluate()
        mAP = results.pop('cityscapes/mAP(%)')
        AP50 = results.pop('cityscapes/AP50(%)')
        self.assertEqual(mAP, 50.0)
        self.assertEqual(AP50, 50.0)

        # except person, others classes ap or ap50 should be nan
        person_ap = results.pop('cityscapes/person_ap(%)')
        person_ap50 = results.pop('cityscapes/person_ap50(%)')
        self.assertEqual(person_ap, 50.0)
        self.assertEqual(person_ap50, 50.0)
        for v in results.values():
            self.assertTrue(math.isnan(v))

        # test format_only
        metric = CityScapesMetric(
            dataset_meta=dataset_metas,
            format_only=True,
            outfile_prefix=osp.join(tmp_dir.name, 'test'),
            seg_prefix=self.seg_prefix,
            keep_results=True,
            keep_gt_json=True,
            classwise=True)

        metric.process({}, data_samples)
        results = metric.evaluate()
        self.assertTrue(osp.exists(f'{osp.join(tmp_dir.name, "test")}'))
        self.assertDictEqual(results, dict())

    def _gen_fake_datasamples(self):
        city = 'lindau'
        os.makedirs(osp.join(self.seg_prefix, city), exist_ok=True)

        sequenceNb = '000000'
        frameNb1 = '000019'
        img_name1 = f'{city}_{sequenceNb}_{frameNb1}_gtFine_instanceIds.png'
        img_path1 = osp.join(self.seg_prefix, city, img_name1)

        masks1 = np.zeros((20, 20), dtype=np.int32)
        masks1[:10, :10] = 24 * 1000
        Image.fromarray(masks1).save(img_path1)

        dummy_mask1 = np.zeros((1, 20, 20), dtype=np.uint8)
        dummy_mask1[:, :10, :10] = 1
        prediction1 = {
            'mask_scores': torch.from_numpy(np.array([1.0])),
            'labels': torch.from_numpy(np.array([0])),
            'masks': torch.from_numpy(dummy_mask1)
        }

        frameNb2 = '000020'
        img_name2 = f'{city}_{sequenceNb}_{frameNb2}_gtFine_instanceIds.png'
        img_path2 = osp.join(self.seg_prefix, city, img_name2)

        masks2 = np.zeros((20, 20), dtype=np.int32)
        masks2[:10, :10] = 24 * 1000 + 1
        Image.fromarray(masks2).save(img_path2)

        dummy_mask2 = np.zeros((1, 20, 20), dtype=np.uint8)
        dummy_mask2[:, :10, :10] = 1
        prediction2 = {
            'mask_scores': torch.from_numpy(np.array([0.98])),
            'labels': torch.from_numpy(np.array([1])),
            'masks': torch.from_numpy(dummy_mask2)
        }

        data_samples = [
            {
                'pred_instances': prediction1,
                'img_path': img_path1
            },
            {
                'pred_instances': prediction2,
                'img_path': img_path2
            },
        ]

        return data_samples
