# Copyright (c) OpenMMLab. All rights reserved.
import unittest

import torch
from mmengine.config import Config

from mmdet.models.seg_heads.panoptic_fusion_heads import MaskFormerFusionHead
from mmdet.structures import DetDataSample


class TestMaskFormerFusionHead(unittest.TestCase):

    def test_loss(self):
        head = MaskFormerFusionHead(num_things_classes=2, num_stuff_classes=2)
        result = head.loss()
        self.assertTrue(not head.with_loss)
        self.assertDictEqual(result, dict())

    def test_predict(self):
        mask_cls_results = torch.rand((2, 10, 5))
        mask_pred_results = torch.rand((2, 10, 32, 32))
        batch_data_samples = [
            DetDataSample(
                metainfo={
                    'batch_input_shape': (32, 32),
                    'img_shape': (32, 30),
                    'ori_shape': (30, 30)
                }),
            DetDataSample(
                metainfo={
                    'batch_input_shape': (32, 32),
                    'img_shape': (32, 30),
                    'ori_shape': (29, 30)
                })
        ]

        # get panoptic and instance segmentation results
        test_cfg = Config(
            dict(
                panoptic_on=True,
                semantic_on=False,
                instance_on=True,
                max_per_image=10,
                object_mask_thr=0.3,
                iou_thr=0.3,
                filter_low_score=False))
        head = MaskFormerFusionHead(
            num_things_classes=2, num_stuff_classes=2, test_cfg=test_cfg)
        results = head.predict(
            mask_cls_results,
            mask_pred_results,
            batch_data_samples,
            rescale=False)
        for i in range(len(results)):
            self.assertEqual(results[i]['pan_results'].sem_seg.shape[-2:],
                             batch_data_samples[i].img_shape)
            self.assertEqual(results[i]['ins_results'].masks.shape[-2:],
                             batch_data_samples[i].img_shape)

        results = head.predict(
            mask_cls_results,
            mask_pred_results,
            batch_data_samples,
            rescale=True)
        for i in range(len(results)):
            self.assertEqual(results[i]['pan_results'].sem_seg.shape[-2:],
                             batch_data_samples[i].ori_shape)
            self.assertEqual(results[i]['ins_results'].masks.shape[-2:],
                             batch_data_samples[i].ori_shape)

        # get empty results
        test_cfg = Config(
            dict(
                panoptic_on=False,
                semantic_on=False,
                instance_on=False,
                max_per_image=10,
                object_mask_thr=0.3,
                iou_thr=0.3,
                filter_low_score=False))
        head = MaskFormerFusionHead(
            num_things_classes=2, num_stuff_classes=2, test_cfg=test_cfg)
        results = head.predict(
            mask_cls_results,
            mask_pred_results,
            batch_data_samples,
            rescale=True)
        for i in range(len(results)):
            self.assertEqual(results[i], dict())

        # semantic segmentation is not supported
        test_cfg = Config(
            dict(
                panoptic_on=False,
                semantic_on=True,
                instance_on=False,
                max_per_image=10,
                object_mask_thr=0.3,
                iou_thr=0.3,
                filter_low_score=False))
        head = MaskFormerFusionHead(
            num_things_classes=2, num_stuff_classes=2, test_cfg=test_cfg)
        with self.assertRaises(AssertionError):
            results = head.predict(
                mask_cls_results,
                mask_pred_results,
                batch_data_samples,
                rescale=True)
