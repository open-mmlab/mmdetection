# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
import tempfile
from unittest import TestCase

import torch
from mmengine.structures import BaseDataElement, InstanceData

from mmdet.evaluation import MOTChallengeMetric
from mmdet.structures import DetDataSample, TrackDataSample


class TestMOTChallengeMetric(TestCase):

    def test_init(self):
        with self.assertRaisesRegex(KeyError, 'metric unknown is not'):
            MOTChallengeMetric(metric='unknown')
        with self.assertRaises(AssertionError):
            MOTChallengeMetric(benchmark='MOT21')

    def __del__(self):
        self.tmp_dir.cleanup()

    @staticmethod
    def _get_predictions_demo():
        instances = [{
            'bbox_label': 0,
            'bbox': [0, 0, 100, 100],
            'ignore_flag': 0,
            'instance_id': 1,
            'mot_conf': 1.0,
            'category_id': 1,
            'visibility': 1.0
        }, {
            'bbox_label': 0,
            'bbox': [0, 0, 100, 100],
            'ignore_flag': 0,
            'instance_id': 2,
            'mot_conf': 1.0,
            'category_id': 1,
            'visibility': 1.0
        }]
        instances_2 = copy.deepcopy(instances)
        sep = os.sep
        pred_instances_data = dict(
            bboxes=torch.tensor([
                [0, 0, 100, 100],
                [0, 0, 100, 40],
            ]),
            instances_id=torch.tensor([1, 2]),
            scores=torch.tensor([1.0, 1.0]))
        pred_instances_data_2 = copy.deepcopy(pred_instances_data)
        pred_instances = InstanceData(**pred_instances_data)
        pred_instances_2 = InstanceData(**pred_instances_data_2)
        img_data_sample = DetDataSample()
        img_data_sample.pred_track_instances = pred_instances
        img_data_sample.instances = instances
        img_data_sample.set_metainfo(
            dict(
                frame_id=0,
                ori_video_length=2,
                video_length=2,
                img_id=1,
                img_path=f'xxx{sep}MOT17-09-DPM{sep}img1{sep}000001.jpg',
            ))
        img_data_sample_2 = DetDataSample()
        img_data_sample_2.pred_track_instances = pred_instances_2
        img_data_sample_2.instances = instances_2
        img_data_sample_2.set_metainfo(
            dict(
                frame_id=1,
                ori_video_length=2,
                video_length=2,
                img_id=2,
                img_path=f'xxx{sep}MOT17-09-DPM{sep}img1{sep}000002.jpg',
            ))
        track_data_sample = TrackDataSample()
        track_data_sample.video_data_samples = [
            img_data_sample, img_data_sample_2
        ]
        # [TrackDataSample]
        predictions = []
        if isinstance(track_data_sample, BaseDataElement):
            predictions.append(track_data_sample.to_dict())
        return predictions

    def _test_evaluate(self, format_only, outfile_predix=None):
        """Test using the metric in the same way as Evaluator."""
        metric = MOTChallengeMetric(
            metric=['HOTA', 'CLEAR', 'Identity'],
            format_only=format_only,
            outfile_prefix=outfile_predix)
        metric.dataset_meta = {'classes': ('pedestrian', )}
        data_batch = dict(input=None, data_samples=None)
        predictions = self._get_predictions_demo()
        metric.process(data_batch, predictions)
        eval_results = metric.evaluate()
        return eval_results

    def test_evaluate(self):
        eval_results = self._test_evaluate(False)
        target = {
            'motchallenge-metric/IDF1': 0.5,
            'motchallenge-metric/MOTA': 0,
            'motchallenge-metric/HOTA': 0.755,
            'motchallenge-metric/IDSW': 0,
        }
        for key in target:
            assert eval_results[key] - target[key] < 1e-3

    def test_evaluate_format_only(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        eval_results = self._test_evaluate(
            True, outfile_predix=self.tmp_dir.name)
        assert eval_results == dict()
