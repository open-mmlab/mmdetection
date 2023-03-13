# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
from mmengine.config import Config
from mmengine.structures import InstanceData

from mmdet.registry import MODELS
from mmdet.testing import demo_tracking_inputs, random_boxes
from mmdet.utils import register_all_modules


def _fake_proposals(img_metas, proposal_len):
    """Create a fake proposal list."""
    results = []
    for i in range(len(img_metas)):
        result = InstanceData(metainfo=img_metas[i])
        proposal = random_boxes(proposal_len, 10).to(device='cpu')
        result.bboxes = proposal
        results.append(result)
    return results


class TestRoITrackHead(TestCase):

    def setUp(self):
        register_all_modules(init_default_scope=True)
        cfg = Config(
            dict(
                roi_extractor=dict(
                    type='SingleRoIExtractor',
                    roi_layer=dict(
                        type='RoIAlign', output_size=7, sampling_ratio=0),
                    out_channels=256,
                    featmap_strides=[4, 8, 16, 32]),
                embed_head=dict(
                    type='RoIEmbedHead',
                    num_fcs=2,
                    roi_feat_size=7,
                    in_channels=256,
                    fc_out_channels=1024),
                train_cfg=dict(
                    assigner=dict(
                        type='MaxIoUAssigner',
                        pos_iou_thr=0.5,
                        neg_iou_thr=0.5,
                        min_pos_iou=0.5,
                        match_low_quality=True,
                        ignore_iof_thr=-1),
                    sampler=dict(
                        type='RandomSampler',
                        num=128,
                        pos_fraction=0.25,
                        neg_pos_ub=-1,
                        add_gt_as_proposals=True),
                    pos_weight=-1,
                )))
        self.track_head = MODELS.build(cfg)

    def test_roi_track_head_loss(self):
        packed_inputs = demo_tracking_inputs(
            batch_size=1,
            key_frame_inds=0,
            num_frames=2,
            image_shapes=[(3, 256, 256)])
        img_metas = [{
            'img_shape': (256, 256, 3),
            'scale_factor': 1,
        }]
        proposal_list = _fake_proposals(img_metas, 10)
        feats = []
        for i in range(len(self.track_head.roi_extractor.featmap_strides)):
            feats.append(
                torch.rand(1, 256, 256 // (2**(i + 2)),
                           256 // (2**(i + 2))).to(device='cpu'))
        key_feats = tuple(feats)
        ref_feats = key_feats
        loss_track = self.track_head.loss(key_feats, ref_feats, proposal_list,
                                          proposal_list,
                                          [packed_inputs['data_samples'][0]])
        assert loss_track['loss_track'] >= 0, 'track loss should be zero'

    def test_roi_track_head_predict(self):
        feats = []
        for i in range(len(self.track_head.roi_extractor.featmap_strides)):
            feats.append(
                torch.rand(1, 256, 256 // (2**(i + 2)),
                           256 // (2**(i + 2))).to(device='cpu'))
        feats = tuple(feats)
        track_feat = self.track_head.predict(
            feats, [torch.Tensor([[10, 10, 20, 20]])])
        assert track_feat.size() == (1, 256)
