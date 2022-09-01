import os.path as osp
import tempfile
from unittest import TestCase

from mmdet.evaluation import CrowdHumanMetric


class TestCrowdHumanMetric(TestCase):

    def setUp(self):
        self.data_root = 'tests/data/crowdhuman_dataset/'
        self.ann_file = 'test_annotation_train.odgt'
        self.tmp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        pass

    def test_init(self):
        ann_file_path = osp.join(self.data_root, self.ann_file)
        with self.assertRaisesRegex(KeyError, 'metric should be one of'):
            CrowdHumanMetric(ann_file=ann_file_path, metric='unknown')

    def test_evaluate(self):

        ann_file_path = osp.join(self.data_root, self.ann_file)
        crowdhuman_metric = CrowdHumanMetric(
            ann_file=ann_file_path,
            classwise=False,
            outfile_prefix=f'{self.tmp_dir.name}/test')

        crowdhuman_metric.process([
            dict(
                inputs=None,
                data_sample={
                    'img_id': 0,
                    'ori_shape': (640, 640)
                })
        ], [dict(pred_instances=crowdhuman_metric)])
