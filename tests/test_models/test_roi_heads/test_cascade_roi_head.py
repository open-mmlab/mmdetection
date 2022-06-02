# Copyright (c) OpenMMLab. All rights reserved.
import unittest
from unittest import TestCase

import torch
from mmengine.data import InstanceData
from parameterized import parameterized

from mmdet.models.roi_heads import StandardRoIHead  # noqa
from mmdet.registry import MODELS
from tests.test_models.test_detectors import demo_mm_inputs
from tests.test_models.test_detectors.utils import get_detector_cfg


def _fake_roi_head(cfg_file):
    """Set a fake roi head config."""
    model = get_detector_cfg(cfg_file)
    roi_head = model.roi_head
    rcnn_train_cfg = model.train_cfg.rcnn if model.train_cfg is not None \
        else None
    roi_head.update(train_cfg=rcnn_train_cfg)
    return roi_head


def _fake_proposals(img_metas, proposal_len):
    """Create a fake proposal list."""
    results = []
    for i in range(len(img_metas)):
        result = InstanceData(metainfo=img_metas[i])
        proposal = torch.randn(proposal_len, 4).to(device='cuda')
        result.bboxes = proposal
        results.append(result)
    return results


class TestStandardRoIHead(TestCase):

    @parameterized.expand(
        ['cascade_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco.py'])
    def test_init(self, cfg_file):
        """Test init standard RoI head."""
        # Normal Cascade Mask R-CNN RoI head
        roi_head_cfg = _fake_roi_head(cfg_file)
        roi_head = MODELS.build(roi_head_cfg)
        assert roi_head.with_bbox
        assert roi_head.with_mask

    @parameterized.expand(
        ['cascade_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco.py'])
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
        roi_head_cfg = _fake_roi_head(cfg_file)
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
        proposal_list = _fake_proposals(img_metas, 100)
        packed_inputs = demo_mm_inputs(
            batch_size=1,
            image_shapes=[(s, s, 3)],
            num_items=[1],
            num_classes=4,
            with_mask=True)
        batch_data_samples = []
        for i in range(len(packed_inputs)):
            batch_data_samples.append(
                packed_inputs[i]['data_sample'].to(device='cuda'))
        out = roi_head.forward_train(feats, proposal_list, batch_data_samples)
        for name, value in out.items():
            if 'loss' in name:
                self.assertGreaterEqual(
                    value.sum(), 0, msg='loss should be non-zero')

        # When there is no truth, the cls loss should be nonzero but
        # there should be no box and mask loss.
        proposal_list = _fake_proposals(img_metas, 100)
        packed_inputs = demo_mm_inputs(
            batch_size=1,
            image_shapes=[(s, s, 3)],
            num_items=[0],
            num_classes=4,
            with_mask=True)
        batch_data_samples = []
        for i in range(len(packed_inputs)):
            batch_data_samples.append(
                packed_inputs[i]['data_sample'].to(device='cuda'))
        out = roi_head.forward_train(feats, proposal_list, batch_data_samples)
        for name, value in out.items():
            if 'loss_cls' in name:
                self.assertGreaterEqual(
                    value.sum(), 0, msg='loss should be non-zero')
            elif 'loss_bbox' in name or 'loss_mask' in name:
                self.assertEqual(value.sum(), 0)
