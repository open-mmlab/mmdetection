# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase
from unittest.mock import MagicMock

import torch
from mmengine.registry import init_default_scope
from parameterized import parameterized

from mmdet.registry import MODELS, TASK_UTILS
from mmdet.testing import demo_track_inputs, get_detector_cfg, random_boxes


class TestStrongSORTTracker(TestCase):

    @classmethod
    def setUpClass(cls):
        init_default_scope('mmdet')
        cls.num_objs = 30

    @parameterized.expand([
        'strongsort/strongsort_yolox_x_8xb4-80e_crowdhuman-mot17halftrain'
        '_test-mot17halfval.py'
    ])
    def test_init(self, cfg_file):
        cfg = get_detector_cfg(cfg_file)
        tracker = MODELS.build(cfg['tracker'])
        tracker.kf = TASK_UTILS.build(cfg['tracker']['motion'])
        tracker.cmc = TASK_UTILS.build(cfg['cmc'])

        bboxes = random_boxes(self.num_objs, 512)
        labels = torch.zeros(self.num_objs)
        scores = torch.ones(self.num_objs)
        ids = torch.arange(self.num_objs)
        tracker.update(
            ids=ids, bboxes=bboxes, scores=scores, labels=labels, frame_ids=0)

        assert tracker.ids == list(ids)
        assert tracker.memo_items == [
            'ids', 'bboxes', 'scores', 'labels', 'frame_ids'
        ]

    @parameterized.expand([
        'strongsort/strongsort_yolox_x_8xb4-80e_crowdhuman-mot17halftrain'
        '_test-mot17halfval.py'
    ])
    def test_track(self, cfg_file):
        img = torch.rand((1, 3, 128, 128))

        cfg = get_detector_cfg(cfg_file)
        tracker = MODELS.build(cfg['tracker'])
        tracker.kf = TASK_UTILS.build(cfg['tracker']['motion'])

        model = MagicMock()
        model.reid = MODELS.build(cfg['reid'])
        model.cmc = TASK_UTILS.build(cfg['cmc'])

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

                pred_track_instances = tracker.track(
                    model=model,
                    img=img,
                    data_sample=img_data_sample,
                    data_preprocessor=cfg['data_preprocessor'])

                bboxes = pred_track_instances.bboxes
                labels = pred_track_instances.labels

                assert bboxes.shape[1] == 4
                assert bboxes.shape[0] == labels.shape[0]
