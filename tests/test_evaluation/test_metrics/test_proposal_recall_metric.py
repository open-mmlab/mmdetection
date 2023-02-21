import tempfile
from unittest import TestCase

import numpy as np
import torch

from mmdet.evaluation import ProposalRecallMetric


class TestCocoMetric(TestCase):

    def _create_dummy_gt(self):
        bboxes = np.array([[50, 60, 70, 80], [100, 120, 130, 150],
                           [150, 160, 190, 200], [250, 260, 350, 360]])
        labels = np.array([0, 0, 1, 0])
        return dict(
            bboxes=torch.from_numpy(bboxes), labels=torch.from_numpy(labels))

    def _create_dummy_results(self):
        bboxes = np.array([[50, 60, 70, 80], [100, 120, 130, 150],
                           [150, 160, 190, 200], [250, 260, 350, 360]])
        scores = np.array([1.0, 0.98, 0.96, 0.95])
        return dict(
            bboxes=torch.from_numpy(bboxes), scores=torch.from_numpy(scores))

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_init(self):
        # test invalid iou_thrs
        with self.assertRaises(TypeError):
            ProposalRecallMetric(iou_thrs={'a', 0.5})

        metric = ProposalRecallMetric(iou_thrs=0.6)
        self.assertTrue(np.array_equal(metric.iou_thrs, np.array([0.6])))

    def test_evaluate(self):
        # create dummy data
        dummy_gt = self._create_dummy_gt()
        dummy_pred = self._create_dummy_results()

        # test single coco dataset evaluation
        proposal_metric = ProposalRecallMetric(
            proposal_nums=(1, 10, 100, 1000))
        proposal_metric.process(
            {}, [dict(gt_instances=dummy_gt, pred_instances=dummy_pred)])
        eval_results = proposal_metric.evaluate()
        target = {
            'AR@1(%)': 25.0,
            'AR@10(%)': 100.0,
            'AR@100(%)': 100.0,
            'AR@1000(%)': 100.0
        }
        self.assertDictEqual(eval_results, target)
