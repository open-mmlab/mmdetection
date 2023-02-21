# Copyright (c) OpenMMLab. All rights reserved.
import unittest

import numpy as np
import torch

from mmdet.datasets import VOCDataset
from mmdet.evaluation import VOCMetric
from mmdet.utils import register_all_modules


class TesVOCMetric(unittest.TestCase):

    def _create_dummy_results(self):
        bboxes = np.array([[48.181, 239.456, 194.984, 371.243],
                           [8, 12, 352, 498]])
        scores = np.array([1.0, 0.98])
        labels = np.array([0, 1])
        return dict(
            bboxes=torch.from_numpy(bboxes),
            scores=torch.from_numpy(scores),
            labels=torch.from_numpy(labels))

    def test_init(self):
        # test invalid iou_thrs
        with self.assertRaises(AssertionError):
            VOCMetric(iou_thrs={'a', 0.5})

        metric = VOCMetric(iou_thrs=0.6)
        self.assertEqual(metric.iou_thrs, [0.6])

    def test_eval(self):
        register_all_modules()
        dataset = VOCDataset(
            data_root='tests/data/VOCdevkit/',
            ann_file='VOC2007/ImageSets/Main/test.txt',
            data_prefix=dict(sub_data_root='VOC2007/'),
            metainfo=dict(classes=('dog', 'person')),
            test_mode=True,
            pipeline=[
                dict(type='LoadAnnotations', with_bbox=True),
                dict(
                    type='PackDetInputs',
                    meta_keys=('img_id', 'img_path', 'instances'))
            ])
        dataset.full_init()
        data_sample = dataset[0]['data_samples'].to_dict()
        data_sample['pred_instances'] = self._create_dummy_results()

        metric = VOCMetric()
        metric.dataset_meta = dataset.metainfo
        metric.process({}, [data_sample])
        results = metric.evaluate()
        targets = {'pascal_voc/mAP@0.5(%)': 100.0, 'pascal_voc/mAP(%)': 100.0}
        self.assertDictEqual(results, targets)

        # test multi-threshold
        data_sample = dataset[0]['data_samples'].to_dict()
        data_sample['pred_instances'] = self._create_dummy_results()
        metric = VOCMetric(iou_thrs=[0.1, 0.5])
        metric.dataset_meta = dataset.metainfo
        metric.process({}, [data_sample])
        results = metric.evaluate()
        targets = {
            'pascal_voc/mAP@0.1(%)': 100.0,
            'pascal_voc/mAP@0.5(%)': 100.0,
            'pascal_voc/mAP(%)': 100.0
        }
        self.assertDictEqual(results, targets)
