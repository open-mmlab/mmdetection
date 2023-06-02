# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData

from mmdet.models.dense_heads import SABLRetinaHead


class TestSABLRetinaHead(TestCase):

    def test_sabl_retina_head(self):
        """Tests sabl retina head loss when truth is empty and non-empty."""
        s = 256
        img_metas = [{
            'img_shape': (s, s),
            'pad_shape': (s, s),
            'scale_factor': [1, 1],
        }]
        train_cfg = ConfigDict(
            dict(
                assigner=dict(
                    type='ApproxMaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.4,
                    min_pos_iou=0.0,
                    ignore_iof_thr=-1),
                allowed_border=-1,
                pos_weight=-1,
                debug=False))
        sabl_retina_head = SABLRetinaHead(
            num_classes=4,
            in_channels=1,
            feat_channels=1,
            stacked_convs=1,
            approx_anchor_generator=dict(
                type='AnchorGenerator',
                octave_base_scale=4,
                scales_per_octave=3,
                ratios=[0.5, 1.0, 2.0],
                strides=[8, 16, 32, 64, 128]),
            square_anchor_generator=dict(
                type='AnchorGenerator',
                ratios=[1.0],
                scales=[4],
                strides=[8, 16, 32, 64, 128]),
            bbox_coder=dict(
                type='BucketingBBoxCoder', num_buckets=14, scale_factor=3.0),
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0),
            loss_bbox_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.5),
            loss_bbox_reg=dict(
                type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.5),
            train_cfg=train_cfg)

        # Fcos head expects a multiple levels of features per image
        feats = (
            torch.rand(1, 1, s // stride[1], s // stride[0])
            for stride in sabl_retina_head.square_anchor_generator.strides)
        outs = sabl_retina_head.forward(feats)

        # Test that empty ground truth encourages the network to
        # predict background
        gt_instances = InstanceData()
        gt_instances.bboxes = torch.empty((0, 4))
        gt_instances.labels = torch.LongTensor([])

        empty_gt_losses = sabl_retina_head.loss_by_feat(
            *outs, [gt_instances], img_metas)
        # When there is no truth, the cls loss should be nonzero but
        # box loss and centerness loss should be zero
        empty_cls_loss = sum(empty_gt_losses['loss_cls']).item()
        empty_box_cls_loss = sum(empty_gt_losses['loss_bbox_cls']).item()
        empty_box_reg_loss = sum(empty_gt_losses['loss_bbox_reg']).item()
        self.assertGreater(empty_cls_loss, 0, 'cls loss should be non-zero')
        self.assertEqual(
            empty_box_cls_loss, 0,
            'there should be no box loss when there are no true boxes')
        self.assertEqual(
            empty_box_reg_loss, 0,
            'there should be no centerness loss when there are no true boxes')

        # When truth is non-empty then all cls, box loss and centerness loss
        # should be nonzero for random inputs
        gt_instances = InstanceData()
        gt_instances.bboxes = torch.Tensor(
            [[23.6667, 23.8757, 238.6326, 151.8874]])
        gt_instances.labels = torch.LongTensor([2])

        one_gt_losses = sabl_retina_head.loss_by_feat(*outs, [gt_instances],
                                                      img_metas)
        onegt_cls_loss = sum(one_gt_losses['loss_cls']).item()
        onegt_box_cls_loss = sum(one_gt_losses['loss_bbox_cls']).item()
        onegt_box_reg_loss = sum(one_gt_losses['loss_bbox_reg']).item()
        self.assertGreater(onegt_cls_loss, 0, 'cls loss should be non-zero')
        self.assertGreater(onegt_box_cls_loss, 0,
                           'box loss should be non-zero')
        self.assertGreater(onegt_box_reg_loss, 0,
                           'centerness loss should be non-zero')

        test_cfg = ConfigDict(
            dict(
                nms_pre=1000,
                min_bbox_size=0,
                score_thr=0.05,
                nms=dict(type='nms', iou_threshold=0.5),
                max_per_img=100))
        # test predict_by_feat
        sabl_retina_head.predict_by_feat(
            *outs, batch_img_metas=img_metas, cfg=test_cfg, rescale=True)
