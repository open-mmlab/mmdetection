import unittest

import torch

from mmdet.models.roi_heads.roi_extractors import SingleRoIExtractor


class TestSingleRoIExtractor(unittest.TestCase):

    def test_forward(self):
        cfg = dict(
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=2),
            out_channels=16,
            featmap_strides=[4, 8, 16, 32])
        roi_extractor = SingleRoIExtractor(**cfg)

        # empty rois
        feats = (torch.rand((1, 16, 200, 336)), )
        rois = torch.empty((0, 5), dtype=torch.float32)
        res = roi_extractor(feats, rois)
        self.assertEqual(len(res), 0)

        # single scale feature
        rois = torch.tensor([[0.0000, 587.8285, 52.1405, 886.2484, 341.5644]])
        res = roi_extractor(feats, rois)
        self.assertEqual(res.shape, (1, 16, 7, 7))

        # multi-scale features
        feats = (
            torch.rand((1, 16, 200, 336)),
            torch.rand((1, 16, 100, 168)),
            torch.rand((1, 16, 50, 84)),
            torch.rand((1, 16, 25, 42)),
        )
        res = roi_extractor(feats, rois)
        self.assertEqual(res.shape, (1, 16, 7, 7))

        res = roi_extractor(feats, rois, roi_scale_factor=2.0)
        self.assertEqual(res.shape, (1, 16, 7, 7))
