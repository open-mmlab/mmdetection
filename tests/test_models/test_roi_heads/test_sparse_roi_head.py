# Copyright (c) OpenMMLab. All rights reserved.
import unittest
from unittest import TestCase

import torch
import torch.nn as nn
from parameterized import parameterized

from mmdet.models.roi_heads import StandardRoIHead  # noqa
from mmdet.registry import MODELS
from mmdet.testing import demo_mm_inputs, demo_mm_proposals, get_roi_head_cfg


class TestCascadeRoIHead(TestCase):

    @parameterized.expand(['queryinst/queryinst_r50_fpn_1x_coco.py'])
    def test_init(self, cfg_file):
        """Test init standard RoI head."""
        # Normal Cascade Mask R-CNN RoI head
        roi_head_cfg = get_roi_head_cfg(cfg_file)
        roi_head = MODELS.build(roi_head_cfg)
        roi_head.init_weights()
        assert roi_head.with_bbox
        assert roi_head.with_mask

    @parameterized.expand(['queryinst/queryinst_r50_fpn_1x_coco.py'])
    def test_cascade_roi_head_loss(self, cfg_file):
        """Tests standard roi head loss when truth is empty and non-empty."""
        if not torch.cuda.is_available():
            # RoI pooling only support in GPU
            return unittest.skip('test requires GPU and torch+cuda')
        s = 256
        img_metas = [{
            'img_shape': (s, s, 3),
            'scale_factor': 1,
        }]
        roi_head_cfg = get_roi_head_cfg(cfg_file)
        roi_head = MODELS.build(roi_head_cfg)
        roi_head = roi_head.cuda()
        feats = []
        for i in range(len(roi_head_cfg.bbox_roi_extractor.featmap_strides)):
            feats.append(
                torch.rand(1, 1, s // (2**(i + 2)),
                           s // (2**(i + 2))).to(device='cuda'))
        feats = tuple(feats)

        # When truth is non-empty then both cls, box, and mask loss
        # should be nonzero for random inputs
        img_shape_list = [(3, s, s) for _ in img_metas]
        proposal_list = demo_mm_proposals(img_shape_list, 100, device='cuda')
        # add import elements into proposal
        init_proposal_features = nn.Embedding(100, 256).cuda().weight.clone()
        for proposal in proposal_list:
            proposal.features = init_proposal_features
            proposal.imgs_whwh = feats[0].new_tensor([[s, s, s,
                                                       s]]).repeat(100, 1)
        batch_data_samples = demo_mm_inputs(
            batch_size=1,
            image_shapes=[(3, s, s)],
            num_items=[1],
            num_classes=4,
            with_mask=True,
            device='cuda')['data_samples']
        out = roi_head.loss(feats, proposal_list, batch_data_samples)
        for name, value in out.items():
            if 'loss' in name:
                self.assertGreaterEqual(
                    value.sum(), 0, msg='loss should be non-zero')

        # When there is no truth, the cls loss should be nonzero but
        # there should be no box and mask loss.
        proposal_list = demo_mm_proposals(img_shape_list, 100, device='cuda')
        # add import elements into proposal
        init_proposal_features = nn.Embedding(100, 256).cuda().weight.clone()
        for proposal in proposal_list:
            proposal.features = init_proposal_features
            proposal.imgs_whwh = feats[0].new_tensor([[s, s, s,
                                                       s]]).repeat(100, 1)
        batch_data_samples = demo_mm_inputs(
            batch_size=1,
            image_shapes=[(3, s, s)],
            num_items=[0],
            num_classes=4,
            with_mask=True,
            device='cuda')['data_samples']
        out = roi_head.loss(feats, proposal_list, batch_data_samples)
        for name, value in out.items():
            if 'loss_cls' in name:
                self.assertGreaterEqual(
                    value.sum(), 0, msg='loss should be non-zero')
            elif 'loss_bbox' in name or 'loss_mask' in name:
                self.assertEqual(value.sum(), 0)
