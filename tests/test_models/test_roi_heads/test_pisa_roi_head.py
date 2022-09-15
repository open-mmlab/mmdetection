# Copyright (c) OpenMMLab. All rights reserved.
import unittest
from unittest import TestCase

import torch
from parameterized import parameterized

from mmdet.registry import MODELS
from mmdet.testing import demo_mm_inputs, demo_mm_proposals, get_roi_head_cfg
from mmdet.utils import register_all_modules


class TestPISARoIHead(TestCase):

    def setUp(self):
        register_all_modules()
        self.roi_head_cfg = get_roi_head_cfg(
            'pisa/faster-rcnn_r50_fpn_pisa_1x_coco.py')

    def test_init(self):
        roi_head = MODELS.build(self.roi_head_cfg)
        self.assertTrue(roi_head.with_bbox)

    @parameterized.expand(['cpu', 'cuda'])
    def test_pisa_roi_head(self, device):
        """Tests trident roi head predict."""
        if not torch.cuda.is_available() and device == 'cuda':
            # RoI pooling only support in GPU
            return unittest.skip('test requires GPU and torch+cuda')
        roi_head = MODELS.build(self.roi_head_cfg)
        roi_head = roi_head.to(device=device)
        s = 256
        feats = []
        for i in range(len(roi_head.bbox_roi_extractor.featmap_strides)):
            feats.append(
                torch.rand(1, 256, s // (2**(i + 2)),
                           s // (2**(i + 2))).to(device=device))

        image_shapes = [(3, s, s)]
        batch_data_samples = demo_mm_inputs(
            batch_size=1,
            image_shapes=image_shapes,
            num_items=[1],
            num_classes=4,
            with_mask=True,
            device=device)['data_samples']
        proposals_list = demo_mm_proposals(
            image_shapes=image_shapes, num_proposals=100, device=device)
        out = roi_head.loss(feats, proposals_list, batch_data_samples)
        loss_cls = out['loss_cls']
        loss_bbox = out['loss_bbox']
        self.assertGreater(loss_cls.sum(), 0, 'cls loss should be non-zero')
        self.assertGreater(loss_bbox.sum(), 0, 'box loss should be non-zero')

        batch_data_samples = demo_mm_inputs(
            batch_size=1,
            image_shapes=image_shapes,
            num_items=[0],
            num_classes=4,
            with_mask=True,
            device=device)['data_samples']
        proposals_list = demo_mm_proposals(
            image_shapes=image_shapes, num_proposals=100, device=device)
        out = roi_head.loss(feats, proposals_list, batch_data_samples)
        empty_cls_loss = out['loss_cls']
        empty_bbox_loss = out['loss_bbox']
        self.assertGreater(empty_cls_loss.sum(), 0,
                           'cls loss should be non-zero')
        self.assertEqual(
            empty_bbox_loss.sum(), 0,
            'there should be no box loss when there are no true boxes')
