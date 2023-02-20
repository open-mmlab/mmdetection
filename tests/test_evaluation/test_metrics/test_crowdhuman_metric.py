import os.path as osp
import tempfile
from unittest import TestCase

import numpy as np
import torch

from mmdet.evaluation import CrowdHumanMetric


class TestCrowdHumanMetric(TestCase):

    def _create_dummy_results(self):
        bboxes = np.array([[1330, 317, 418, 1338], [792, 24, 723, 2017],
                           [693, 291, 307, 894], [522, 290, 285, 826],
                           [728, 336, 175, 602], [92, 337, 267, 681]])
        bboxes[:, 2:4] += bboxes[:, 0:2]
        scores = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        return dict(
            bboxes=torch.from_numpy(bboxes), scores=torch.from_numpy(scores))

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.ann_file_path = \
            'tests/data/crowdhuman_dataset/test_annotation_train.odgt',

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_init(self):
        with self.assertRaisesRegex(KeyError, 'metric should be one of'):
            CrowdHumanMetric(ann_file=self.ann_file_path[0], metric='unknown')

    def test_evaluate(self):
        # create dummy data
        dummy_pred = self._create_dummy_results()

        crowdhuman_metric = CrowdHumanMetric(
            ann_file=self.ann_file_path[0],
            outfile_prefix=f'{self.tmp_dir.name}/test')
        crowdhuman_metric.process({}, [
            dict(
                pred_instances=dummy_pred,
                img_id='283554,35288000868e92d4',
                ori_shape=(1640, 1640))
        ])
        eval_results = crowdhuman_metric.evaluate(size=1)
        target = {
            'crowd_human/mAP': 0.8333,
            'crowd_human/mMR': 0.0,
            'crowd_human/JI': 1.0
        }
        self.assertDictEqual(eval_results, target)
        self.assertTrue(osp.isfile(osp.join(self.tmp_dir.name, 'test.json')))
