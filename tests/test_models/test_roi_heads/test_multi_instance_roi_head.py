# Copyright (c) OpenMMLab. All rights reserved.
import unittest
from unittest import TestCase

import torch
from mmengine.config import Config

from mmdet.registry import MODELS
from mmdet.testing import demo_mm_inputs, demo_mm_proposals
from mmdet.utils import register_all_modules

register_all_modules()


def _fake_roi_head():
    """Set a fake roi head config."""

    roi_head = Config(
        dict(
            type='MultiInstanceRoIHead',
            bbox_roi_extractor=dict(
                type='SingleRoIExtractor',
                roi_layer=dict(
                    type='RoIAlign',
                    output_size=7,
                    sampling_ratio=-1,
                    aligned=True,
                    use_torchvision=True),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32]),
            bbox_head=dict(
                type='MultiInstanceBBoxHead',
                with_refine=False,
                num_shared_fcs=2,
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=False,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    loss_weight=1.0,
                    use_sigmoid=False,
                    reduction='none'),
                loss_bbox=dict(
                    type='SmoothL1Loss', loss_weight=1.0, reduction='none')),
            train_cfg=dict(
                assigner=dict(
                    type='MultiInstanceAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.3,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='MultiInsRandomSampler',
                    num=512,
                    pos_fraction=0.5,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=False),
                pos_weight=-1,
                debug=False),
            test_cfg=dict(
                nms=dict(iou_threshold=0.5), score_thr=0.01, max_per_img=500)))

    return roi_head


class TestMultiInstanceRoIHead(TestCase):

    def test_init(self):
        """Test init multi instance RoI head."""
        roi_head_cfg = _fake_roi_head()
        roi_head = MODELS.build(roi_head_cfg)
        self.assertTrue(roi_head.with_bbox)

    def test_standard_roi_head_loss(self):
        """Tests multi instance roi head loss when truth is empty and non-
        empty."""
        if not torch.cuda.is_available():
            # RoI pooling only support in GPU
            return unittest.skip('test requires GPU and torch+cuda')
        s = 256
        roi_head_cfg = _fake_roi_head()
        roi_head = MODELS.build(roi_head_cfg)
        roi_head = roi_head.cuda()
        feats = []
        for i in range(len(roi_head.bbox_roi_extractor.featmap_strides)):
            feats.append(
                torch.rand(1, 1, s // (2**(i + 2)),
                           s // (2**(i + 2))).to(device='cuda'))
        feats = tuple(feats)

        # When truth is non-empty then emd loss should be nonzero for
        # random inputs
        image_shapes = [(3, s, s)]
        batch_data_samples = demo_mm_inputs(
            batch_size=1,
            image_shapes=image_shapes,
            num_items=[1],
            num_classes=4,
            with_mask=False,
            device='cuda')['data_samples']
        proposals_list = demo_mm_proposals(
            image_shapes=image_shapes, num_proposals=100, device='cuda')

        out = roi_head.loss(feats, proposals_list, batch_data_samples)
        loss = out['loss_rcnn_emd']
        self.assertGreater(loss.sum(), 0, 'loss should be non-zero')

        # When there is no truth, the emd loss should be zero.
        batch_data_samples = demo_mm_inputs(
            batch_size=1,
            image_shapes=image_shapes,
            num_items=[0],
            num_classes=4,
            with_mask=True,
            device='cuda')['data_samples']
        proposals_list = demo_mm_proposals(
            image_shapes=image_shapes, num_proposals=100, device='cuda')
        out = roi_head.loss(feats, proposals_list, batch_data_samples)
        empty_loss = out['loss_rcnn_emd']
        self.assertEqual(
            empty_loss.sum(), 0,
            'there should be no emd loss when there are no true boxes')
