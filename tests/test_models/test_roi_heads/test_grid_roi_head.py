# Copyright (c) OpenMMLab. All rights reserved.
import unittest
from unittest import TestCase

import torch
from parameterized import parameterized

from mmdet.registry import MODELS
from mmdet.testing import demo_mm_inputs, demo_mm_proposals, get_roi_head_cfg
from mmdet.utils import register_all_modules


class TestGridRoIHead(TestCase):

    def setUp(self):
        register_all_modules()
        self.roi_head_cfg = get_roi_head_cfg(
            'grid_rcnn/grid-rcnn_r50_fpn_gn-head_2x_coco.py')

    def test_init(self):
        roi_head = MODELS.build(self.roi_head_cfg)
        self.assertTrue(roi_head.with_bbox)

    @parameterized.expand(['cpu', 'cuda'])
    def test_grid_roi_head_loss(self, device):
        """Tests trident roi head predict."""
        if device == 'cuda':
            if not torch.cuda.is_available():
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
        loss_grid = out['loss_grid']
        self.assertGreater(loss_cls.sum(), 0, 'cls loss should be non-zero')
        self.assertGreater(loss_grid.sum(), 0, 'grid loss should be non-zero')

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
        self.assertGreater(empty_cls_loss.sum(), 0,
                           'cls loss should be non-zero')
        self.assertNotIn(
            'loss_grid', out,
            'grid loss should be passed when there are no true boxes')

    @parameterized.expand(['cpu', 'cuda'])
    def test_grid_roi_head_predict(self, device):
        """Tests trident roi head predict."""
        if device == 'cuda':
            if not torch.cuda.is_available():
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
            num_items=[0],
            num_classes=4,
            with_mask=True,
            device=device)['data_samples']
        proposals_list = demo_mm_proposals(
            image_shapes=image_shapes, num_proposals=100, device=device)
        roi_head.predict(feats, proposals_list, batch_data_samples)

    @parameterized.expand(['cpu', 'cuda'])
    def test_grid_roi_head_forward(self, device):
        """Tests trident roi head forward."""
        if device == 'cuda':
            if not torch.cuda.is_available():
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
        proposals_list = demo_mm_proposals(
            image_shapes=image_shapes, num_proposals=100, device=device)
        roi_head.forward(feats, proposals_list)
