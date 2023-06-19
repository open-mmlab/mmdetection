# Copyright (c) OpenMMLab. All rights reserved.
import os
import tempfile
from unittest import TestCase

import numpy as np
import pycocotools.mask as mask_util
import torch
from mmengine.registry import init_default_scope
from mmengine.structures import BaseDataElement, InstanceData

from mmdet.registry import METRICS
from mmdet.structures import DetDataSample, TrackDataSample


class TestYouTubeVISMetric(TestCase):

    @classmethod
    def setUpClass(cls):
        init_default_scope('mmdet')

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp_dir.cleanup()

    def _create_dummy_results(self, track_id):
        bboxes = np.array([[100, 100, 150, 150]])
        scores = np.array([1.0])
        labels = np.array([0])
        instance_id = np.array([track_id])
        dummy_mask = np.zeros((1, 720, 1280), dtype=np.uint8)
        dummy_mask[:, 100:150, 100:150] = 1
        return dict(
            bboxes=torch.from_numpy(bboxes),
            scores=torch.from_numpy(scores),
            labels=torch.from_numpy(labels),
            instances_id=torch.from_numpy(instance_id),
            masks=torch.from_numpy(dummy_mask))

    def test_format_only(self):
        outfile_prefix = f'{self.tmp_dir.name}/result'
        vis_metric = METRICS.build(
            dict(
                type='YouTubeVISMetric',
                format_only=True,
                outfile_prefix=outfile_prefix,
            ))
        dummy_pred = self._create_dummy_results(track_id=0)
        dummy_mask = np.zeros((720, 1280), order='F', dtype=np.uint8)
        dummy_mask[100:150, 100:150] = 1
        rle_mask = mask_util.encode(dummy_mask)
        rle_mask['counts'] = rle_mask['counts'].decode('utf-8')
        instances = [{
            'bbox_label': 0,
            'bbox': [100, 100, 150, 150],
            'ignore_flag': 0,
            'instance_id': 1,
            'mask': rle_mask,
        }]
        vis_metric.dataset_meta = dict(classes=['car', 'train'])
        data_batch = dict(inputs=None, data_samples=None)
        gt_insatnce = InstanceData(**dummy_pred)
        img_data_sample = DetDataSample()
        img_data_sample.pred_track_instances = gt_insatnce
        img_data_sample.set_metainfo(
            dict(
                img_id=0,
                video_id=1,
                ori_video_length=1,
                ori_shape=(720, 1280),
                instances=instances))
        track_data_sample = TrackDataSample()
        track_data_sample.video_data_samples = [img_data_sample]
        predictions = []
        if isinstance(track_data_sample, BaseDataElement):
            predictions.append(track_data_sample.to_dict())
        vis_metric.process(data_batch, predictions)
        vis_metric.evaluate(size=1)
        assert os.path.exists(f'{outfile_prefix}.json')
        assert os.path.exists(f'{outfile_prefix}.submission_file.zip')

    def test_evaluate(self):
        """Test using the metric in the same way as Evaluator."""
        dummy_pred_1 = self._create_dummy_results(track_id=1)
        dummy_pred_2 = self._create_dummy_results(track_id=1)
        dummy_pred_3 = self._create_dummy_results(track_id=2)

        dummy_mask = np.zeros((720, 1280), order='F', dtype=np.uint8)
        dummy_mask[100:150, 100:150] = 1
        rle_mask = mask_util.encode(dummy_mask)
        rle_mask['counts'] = rle_mask['counts'].decode('utf-8')
        instances_1 = [{
            'bbox_label': 0,
            'bbox': [100, 100, 150, 150],
            'ignore_flag': 0,
            'instance_id': 1,
            'mask': rle_mask,
        }]
        instances_2 = [{
            'bbox_label': 0,
            'bbox': [100, 100, 150, 150],
            'ignore_flag': 0,
            'instance_id': 2,
            'mask': rle_mask,
        }]
        vis_metric = METRICS.build(
            dict(
                type='YouTubeVISMetric',
                outfile_prefix=f'{self.tmp_dir.name}/test',
            ))

        vis_metric.dataset_meta = dict(classes=['car', 'train'])
        data_batch = dict(inputs=None, data_samples=None)
        gt_insatnce = InstanceData(**dummy_pred_1)
        img_data_sample = DetDataSample()
        img_data_sample.pred_track_instances = gt_insatnce
        img_data_sample.set_metainfo(
            dict(
                img_id=1,
                video_id=1,
                ori_video_length=2,
                ori_shape=(720, 1280),
                instances=instances_1))
        gt_insatnce_2 = InstanceData(**dummy_pred_2)
        img_data_sample_2 = DetDataSample()
        img_data_sample_2.pred_track_instances = gt_insatnce_2
        img_data_sample_2.set_metainfo(
            dict(
                img_id=2,
                video_id=1,
                ori_video_length=2,
                ori_shape=(720, 1280),
                instances=instances_1))
        track_data_sample = TrackDataSample()
        track_data_sample.video_data_samples = [
            img_data_sample, img_data_sample_2
        ]
        predictions = []
        if isinstance(track_data_sample, BaseDataElement):
            predictions.append(track_data_sample.to_dict())
        vis_metric.process(data_batch, predictions)

        gt_insatnce = InstanceData(**dummy_pred_3)
        img_data_sample = DetDataSample()
        img_data_sample.pred_track_instances = gt_insatnce
        img_data_sample.set_metainfo(
            dict(
                img_id=3,
                video_id=2,
                ori_video_length=1,
                ori_shape=(720, 1280),
                instances=instances_2))
        track_data_sample = TrackDataSample()
        track_data_sample.video_data_samples = [img_data_sample]
        predictions = []
        if isinstance(track_data_sample, BaseDataElement):
            predictions.append(track_data_sample.to_dict())
        vis_metric.process(data_batch, predictions)

        eval_results = vis_metric.evaluate(size=3)
        target = {
            'youtube_vis/segm_mAP': 1.0,
            'youtube_vis/segm_mAP_50': 1.0,
            'youtube_vis/segm_mAP_75': 1.0,
            'youtube_vis/segm_mAP_s': 1.0,
            'youtube_vis/segm_mAP_m': -1.0,
            'youtube_vis/segm_mAP_l': -1.0,
        }
        self.assertDictEqual(eval_results, target)
