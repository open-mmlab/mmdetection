# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData

from mmdet.models.dense_heads import CascadeRPNHead
from mmdet.structures import DetDataSample

rpn_weight = 0.7
cascade_rpn_config = ConfigDict(
    dict(
        num_stages=2,
        num_classes=1,
        stages=[
            dict(
                type='StageCascadeRPNHead',
                in_channels=1,
                feat_channels=1,
                anchor_generator=dict(
                    type='AnchorGenerator',
                    scales=[8],
                    ratios=[1.0],
                    strides=[4, 8, 16, 32, 64]),
                adapt_cfg=dict(type='dilation', dilation=3),
                bridged_feature=True,
                with_cls=False,
                reg_decoded_bbox=True,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=(.0, .0, .0, .0),
                    target_stds=(0.1, 0.1, 0.5, 0.5)),
                loss_bbox=dict(
                    type='IoULoss', linear=True,
                    loss_weight=10.0 * rpn_weight)),
            dict(
                type='StageCascadeRPNHead',
                in_channels=1,
                feat_channels=1,
                adapt_cfg=dict(type='offset'),
                bridged_feature=False,
                with_cls=True,
                reg_decoded_bbox=True,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=(.0, .0, .0, .0),
                    target_stds=(0.05, 0.05, 0.1, 0.1)),
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=True,
                    loss_weight=1.0 * rpn_weight),
                loss_bbox=dict(
                    type='IoULoss', linear=True,
                    loss_weight=10.0 * rpn_weight))
        ],
        train_cfg=[
            dict(
                assigner=dict(
                    type='RegionAssigner', center_ratio=0.2, ignore_ratio=0.5),
                allowed_border=-1,
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.7,
                    min_pos_iou=0.3,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=256,
                    pos_fraction=0.5,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=False),
                allowed_border=-1,
                pos_weight=-1,
                debug=False)
        ],
        test_cfg=dict(max_per_img=300, nms=dict(iou_threshold=0.8))))


class TestStageCascadeRPNHead(TestCase):

    def test_cascade_rpn_head_loss(self):
        """Tests cascade rpn head loss when truth is empty and non-empty."""
        cascade_rpn_head = CascadeRPNHead(**cascade_rpn_config)

        s = 256
        feats = [
            torch.rand(1, 1, s // stride[1], s // stride[0])
            for stride in cascade_rpn_head.stages[0].prior_generator.strides
        ]
        img_metas = {
            'img_shape': (s, s),
            'pad_shape': (s, s),
            'scale_factor': 1,
        }
        sample = DetDataSample()
        sample.set_metainfo(img_metas)

        # Test that empty ground truth encourages the network to
        # predict background
        gt_instances = InstanceData()
        gt_instances.bboxes = torch.empty((0, 4))
        gt_instances.labels = torch.LongTensor([])
        sample.gt_instances = gt_instances

        empty_gt_losses = cascade_rpn_head.loss(feats, [sample])
        for key, loss in empty_gt_losses.items():
            loss = sum(loss)
            if 'cls' in key:
                self.assertGreater(loss.item(), 0,
                                   'cls loss should be non-zero')
            elif 'reg' in key:
                self.assertEqual(
                    loss.item(), 0,
                    'there should be no reg loss when no ground true boxes')

        # When truth is non-empty then all cls, box loss and centerness loss
        # should be nonzero for random inputs
        gt_instances = InstanceData()
        gt_instances.bboxes = torch.Tensor(
            [[23.6667, 23.8757, 238.6326, 151.8874]])
        gt_instances.labels = torch.LongTensor([0])
        sample.gt_instances = gt_instances

        one_gt_losses = cascade_rpn_head.loss(feats, [sample])
        for loss in one_gt_losses.values():
            loss = sum(loss)
            self.assertGreater(
                loss.item(), 0,
                'cls loss, or box loss, or iou loss should be non-zero')

    def test_cascade_rpn_head_loss_and_predict(self):
        """Tests cascade rpn head loss and predict function."""
        cascade_rpn_head = CascadeRPNHead(**cascade_rpn_config)

        s = 256
        feats = [
            torch.rand(1, 1, s // stride[1], s // stride[0])
            for stride in cascade_rpn_head.stages[0].prior_generator.strides
        ]
        img_metas = {
            'img_shape': (s, s),
            'pad_shape': (s, s),
            'scale_factor': 1,
        }
        sample = DetDataSample()
        sample.set_metainfo(img_metas)

        gt_instances = InstanceData()
        gt_instances.bboxes = torch.empty((0, 4))
        gt_instances.labels = torch.LongTensor([])
        sample.gt_instances = gt_instances
        proposal_cfg = ConfigDict(
            dict(max_per_img=300, nms=dict(iou_threshold=0.8)))

        cascade_rpn_head.loss_and_predict(feats, [sample], proposal_cfg)

    def test_cascade_rpn_head_predict(self):
        """Tests cascade rpn head predict function."""
        cascade_rpn_head = CascadeRPNHead(**cascade_rpn_config)

        s = 256
        feats = [
            torch.rand(1, 1, s // stride[1], s // stride[0])
            for stride in cascade_rpn_head.stages[0].prior_generator.strides
        ]
        img_metas = {
            'img_shape': (s, s),
            'pad_shape': (s, s),
            'scale_factor': 1,
        }
        sample = DetDataSample()
        sample.set_metainfo(img_metas)

        cascade_rpn_head.predict(feats, [sample])
