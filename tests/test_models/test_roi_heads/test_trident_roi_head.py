# Copyright (c) OpenMMLab. All rights reserved.
import copy
import unittest
from unittest import TestCase

import torch

from mmdet.registry import MODELS
from mmdet.testing import demo_mm_inputs, demo_mm_proposals, get_roi_head_cfg
from mmdet.utils import register_all_modules


class TestTridentRoIHead(TestCase):

    def setUp(self):
        register_all_modules()
        self.roi_head_cfg = get_roi_head_cfg(
            'tridentnet/tridentnet_r50-caffe_1x_coco.py')

    def test_init(self):

        roi_head = MODELS.build(self.roi_head_cfg)
        self.assertTrue(roi_head.with_bbox)
        self.assertTrue(roi_head.with_shared_head)

    def test_trident_roi_head_predict(self):
        """Tests trident roi head predict."""
        if not torch.cuda.is_available():
            # RoI pooling only support in GPU
            return unittest.skip('test requires GPU and torch+cuda')

        roi_head_cfg = copy.deepcopy(self.roi_head_cfg)
        roi_head = MODELS.build(roi_head_cfg)
        roi_head = roi_head.cuda()
        s = 256
        feats = []
        for i in range(len(roi_head.bbox_roi_extractor.featmap_strides)):
            feats.append(
                torch.rand(1, 1024, s // (2**(i + 2)),
                           s // (2**(i + 2))).to(device='cuda'))

        image_shapes = [(3, s, s)]
        batch_data_samples = demo_mm_inputs(
            batch_size=1,
            image_shapes=image_shapes,
            num_items=[0],
            num_classes=4,
            with_mask=True,
            device='cuda')['data_samples']
        proposals_list = demo_mm_proposals(
            image_shapes=image_shapes, num_proposals=100, device='cuda')
        # When `test_branch_idx == 1`
        roi_head.predict(feats, proposals_list, batch_data_samples)
        # When `test_branch_idx == -1`
        roi_head_cfg.test_branch_idx = -1
        roi_head = MODELS.build(roi_head_cfg)
        roi_head = roi_head.cuda()
        roi_head.predict(feats, proposals_list, batch_data_samples)
