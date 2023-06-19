# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch

from mmdet.registry import MODELS, TASK_UTILS
from mmdet.testing import demo_track_inputs
from mmdet.utils import register_all_modules


class TestByteTracker(TestCase):

    @classmethod
    def setUpClass(cls):
        register_all_modules(init_default_scope=True)
        cfg = dict(
            type='OCSORTTracker',
            motion=dict(type='KalmanFilter'),
            obj_score_thr=0.3,
            init_track_thr=0.7,
            weight_iou_with_det_scores=True,
            match_iou_thr=0.3,
            num_tentatives=3,
            vel_consist_weight=0.2,
            vel_delta_t=3,
            num_frames_retain=30)
        cls.tracker = MODELS.build(cfg)
        cls.tracker.kf = TASK_UTILS.build(dict(type='KalmanFilter'))
        cls.num_frames_retain = cfg['num_frames_retain']
        cls.num_objs = 30

    def test_track(self):

        with torch.no_grad():
            packed_inputs = demo_track_inputs(batch_size=1, num_frames=2)
            track_data_sample = packed_inputs['data_samples'][0]
            video_len = len(track_data_sample)
            for frame_id in range(video_len):
                img_data_sample = track_data_sample[frame_id]
                img_data_sample.pred_instances = \
                    img_data_sample.gt_instances.clone()
                # add fake scores
                scores = torch.ones(len(img_data_sample.gt_instances.bboxes))
                img_data_sample.pred_instances.scores = torch.FloatTensor(
                    scores)

                pred_track_instances = self.tracker.track(
                    data_sample=img_data_sample)

                bboxes = pred_track_instances.bboxes
                labels = pred_track_instances.labels

                assert bboxes.shape[1] == 4
                assert bboxes.shape[0] == labels.shape[0]
