# Copyright (c) OpenMMLab. All rights reserved.
import unittest

import numpy as np
import torch

from mmdet.datasets import OpenImagesDataset
from mmdet.evaluation import OpenImagesMetric
from mmdet.utils import register_all_modules


class TestOpenImagesMetric(unittest.TestCase):

    def _create_dummy_results(self):
        bboxes = np.array([[23.2172, 31.7541, 987.3413, 357.8443],
                           [100, 120, 130, 150], [150, 160, 190, 200],
                           [250, 260, 350, 360]])
        scores = np.array([1.0, 0.98, 0.96, 0.95])
        labels = np.array([0, 0, 0, 0])
        return dict(
            bboxes=torch.from_numpy(bboxes),
            scores=torch.from_numpy(scores),
            labels=torch.from_numpy(labels))

    def test_init(self):
        # test invalid iou_thrs
        with self.assertRaises(AssertionError):
            OpenImagesMetric(iou_thrs={'a', 0.5}, ioa_thrs={'b', 0.5})
        # test ioa and iou_thrs length not equal
        with self.assertRaises(AssertionError):
            OpenImagesMetric(iou_thrs=[0.5, 0.75], ioa_thrs=[0.5])

        metric = OpenImagesMetric(iou_thrs=0.6)
        self.assertEqual(metric.iou_thrs, [0.6])

    def test_eval(self):
        register_all_modules()
        dataset = OpenImagesDataset(
            data_root='tests/data/OpenImages/',
            ann_file='annotations/oidv6-train-annotations-bbox.csv',
            data_prefix=dict(img='OpenImages/train/'),
            label_file='annotations/class-descriptions-boxable.csv',
            hierarchy_file='annotations/bbox_labels_600_hierarchy.json',
            meta_file='annotations/image-metas.pkl',
            pipeline=[
                dict(type='LoadAnnotations', with_bbox=True),
                dict(
                    type='PackDetInputs',
                    meta_keys=('img_id', 'img_path', 'instances'))
            ])
        dataset.full_init()
        data_sample = dataset[0]['data_samples'].to_dict()
        data_sample['pred_instances'] = self._create_dummy_results()

        metric = OpenImagesMetric()
        metric.dataset_meta = dataset.metainfo
        metric.process({}, [data_sample])
        results = metric.evaluate(size=len(dataset))
        targets = {'openimages/AP50': 1.0, 'openimages/mAP': 1.0}
        self.assertDictEqual(results, targets)

        # test multi-threshold
        metric = OpenImagesMetric(iou_thrs=[0.1, 0.5], ioa_thrs=[0.1, 0.5])
        metric.dataset_meta = dataset.metainfo
        metric.process({}, [data_sample])
        results = metric.evaluate(size=len(dataset))
        targets = {
            'openimages/AP10': 1.0,
            'openimages/AP50': 1.0,
            'openimages/mAP': 1.0
        }
        self.assertDictEqual(results, targets)
