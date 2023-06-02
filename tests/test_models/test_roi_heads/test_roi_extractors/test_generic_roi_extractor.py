import unittest

import torch

from mmdet.models.roi_heads.roi_extractors import GenericRoIExtractor


class TestGenericRoIExtractor(unittest.TestCase):

    def test_init(self):
        with self.assertRaises(AssertionError):
            GenericRoIExtractor(
                aggregation='other',
                roi_layer=dict(
                    type='RoIAlign', output_size=7, sampling_ratio=2),
                out_channels=16,
                featmap_strides=[4, 8, 16, 32])

        roi_extractor = GenericRoIExtractor(
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=2),
            out_channels=16,
            featmap_strides=[4, 8, 16, 32])
        self.assertFalse(roi_extractor.with_pre)
        self.assertFalse(roi_extractor.with_post)

    def test_forward(self):
        # test with pre/post
        cfg = dict(
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=2),
            out_channels=16,
            featmap_strides=[4, 8, 16, 32],
            pre_cfg=dict(
                type='ConvModule',
                in_channels=16,
                out_channels=16,
                kernel_size=5,
                padding=2,
                inplace=False),
            post_cfg=dict(
                type='ConvModule',
                in_channels=16,
                out_channels=16,
                kernel_size=5,
                padding=2,
                inplace=False))
        roi_extractor = GenericRoIExtractor(**cfg)

        # empty rois
        feats = (
            torch.rand((1, 16, 200, 336)),
            torch.rand((1, 16, 100, 168)),
        )
        rois = torch.empty((0, 5), dtype=torch.float32)
        res = roi_extractor(feats, rois)
        self.assertEqual(len(res), 0)

        # single scale feature
        rois = torch.tensor([[0.0000, 587.8285, 52.1405, 886.2484, 341.5644]])
        feats = (torch.rand((1, 16, 200, 336)), )
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

        # test w.o. pre/post concat
        cfg = dict(
            aggregation='concat',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=2),
            out_channels=16 * 4,
            featmap_strides=[4, 8, 16, 32])

        roi_extractor = GenericRoIExtractor(**cfg)
        res = roi_extractor(feats, rois)
        self.assertEqual(res.shape, (1, 64, 7, 7))

        # test concat channels number
        cfg = dict(
            aggregation='concat',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=2),
            out_channels=256 * 5,  # 256*5 != 256*4
            featmap_strides=[4, 8, 16, 32])

        roi_extractor = GenericRoIExtractor(**cfg)

        feats = (
            torch.rand((1, 256, 200, 336)),
            torch.rand((1, 256, 100, 168)),
            torch.rand((1, 256, 50, 84)),
            torch.rand((1, 256, 25, 42)),
        )
        # out_channels does not sum of feat channels
        with self.assertRaises(AssertionError):
            roi_extractor(feats, rois)
