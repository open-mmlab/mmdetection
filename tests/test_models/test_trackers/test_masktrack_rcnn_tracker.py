# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
from mmengine.registry import init_default_scope
from parameterized import parameterized

from mmdet.registry import MODELS
from mmdet.testing import demo_track_inputs, get_detector_cfg, random_boxes


class TestMaskTrackRCNNTracker(TestCase):

    @classmethod
    def setUpClass(cls):
        init_default_scope('mmdet')
        tracker_cfg = dict(
            type='MaskTrackRCNNTracker',
            match_weights=dict(det_score=1.0, iou=2.0, det_label=10.0),
            num_frames_retain=20)
        cls.tracker = MODELS.build(tracker_cfg)
        cls.num_objs = 5

    def test_get_match_score(self):
        bboxes = random_boxes(self.num_objs, 64)
        labels = torch.arange(self.num_objs)
        scores = torch.arange(self.num_objs, dtype=torch.float32)
        similarity_logits = torch.randn(self.num_objs, self.num_objs + 1)

        match_score = self.tracker.get_match_score(bboxes, labels, scores,
                                                   bboxes, labels,
                                                   similarity_logits)
        assert match_score.size() == similarity_logits.size()

    @parameterized.expand([
        'masktrack_rcnn/masktrack-rcnn_mask-rcnn_r50_fpn_8xb1-12e_youtubevis2019.py'  # noqa: E501
    ])
    def test_track(self, cfg_file):
        _model = get_detector_cfg(cfg_file)
        # _scope_ will be popped after build
        model = MODELS.build(_model)

        packed_inputs = demo_track_inputs(
            batch_size=1, num_frames=2, with_mask=True)
        track_data_sample = packed_inputs['data_samples'][0]
        imgs = packed_inputs['inputs'][0]
        video_len = len(track_data_sample)
        for frame_id in range(video_len):
            img_data_sample = track_data_sample[frame_id]
            single_image = imgs[frame_id]
            img_data_sample.pred_instances = \
                img_data_sample.gt_instances.clone()
            # add fake scores
            scores = torch.ones(len(img_data_sample.pred_instances.bboxes))
            img_data_sample.pred_instances.scores = torch.FloatTensor(scores)
            feats = []
            for i in range(
                    len(model.track_head.roi_extractor.featmap_strides)):
                feats.append(
                    torch.rand(1, 256, 256 // (2**(i + 2)),
                               256 // (2**(i + 2))).to(device='cpu'))
            pred_track_instances = self.tracker.track(
                model=model,
                img=single_image,
                feats=tuple(feats),
                data_sample=img_data_sample)

            bboxes = pred_track_instances.bboxes
            labels = pred_track_instances.labels
            ids = pred_track_instances.instances_id

            assert bboxes.shape[1] == 4
            assert bboxes.shape[0] == labels.shape[0]
            assert bboxes.shape[0] == ids.shape[0]
