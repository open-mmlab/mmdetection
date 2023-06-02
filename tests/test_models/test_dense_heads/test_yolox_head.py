# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmengine.config import Config
from mmengine.model import bias_init_with_prob
from mmengine.structures import InstanceData
from mmengine.testing import assert_allclose

from mmdet.models.dense_heads import YOLOXHead


class TestYOLOXHead(TestCase):

    def test_init_weights(self):
        head = YOLOXHead(
            num_classes=4, in_channels=1, stacked_convs=1, use_depthwise=False)
        head.init_weights()
        bias_init = bias_init_with_prob(0.01)
        for conv_cls, conv_obj in zip(head.multi_level_conv_cls,
                                      head.multi_level_conv_obj):
            assert_allclose(conv_cls.bias.data,
                            torch.ones_like(conv_cls.bias.data) * bias_init)
            assert_allclose(conv_obj.bias.data,
                            torch.ones_like(conv_obj.bias.data) * bias_init)

    def test_predict_by_feat(self):
        s = 256
        img_metas = [{
            'img_shape': (s, s, 3),
            'scale_factor': (1.0, 1.0),
        }]
        test_cfg = Config(
            dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.65)))
        head = YOLOXHead(
            num_classes=4,
            in_channels=1,
            stacked_convs=1,
            use_depthwise=False,
            test_cfg=test_cfg)
        feat = [
            torch.rand(1, 1, s // feat_size, s // feat_size)
            for feat_size in [4, 8, 16]
        ]
        cls_scores, bbox_preds, objectnesses = head.forward(feat)
        head.predict_by_feat(
            cls_scores,
            bbox_preds,
            objectnesses,
            img_metas,
            cfg=test_cfg,
            rescale=True,
            with_nms=True)
        head.predict_by_feat(
            cls_scores,
            bbox_preds,
            objectnesses,
            img_metas,
            cfg=test_cfg,
            rescale=False,
            with_nms=False)

    def test_loss_by_feat(self):
        s = 256
        img_metas = [{
            'img_shape': (s, s, 3),
            'scale_factor': 1,
        }]
        train_cfg = Config(
            dict(
                assigner=dict(
                    type='SimOTAAssigner',
                    center_radius=2.5,
                    candidate_topk=10,
                    iou_weight=3.0,
                    cls_weight=1.0)))
        head = YOLOXHead(
            num_classes=4,
            in_channels=1,
            stacked_convs=1,
            use_depthwise=False,
            train_cfg=train_cfg)
        assert not head.use_l1
        assert isinstance(head.multi_level_cls_convs[0][0], ConvModule)

        feat = [
            torch.rand(1, 1, s // feat_size, s // feat_size)
            for feat_size in [4, 8, 16]
        ]
        cls_scores, bbox_preds, objectnesses = head.forward(feat)

        # Test that empty ground truth encourages the network to predict
        # background
        gt_instances = InstanceData(
            bboxes=torch.empty((0, 4)), labels=torch.LongTensor([]))

        empty_gt_losses = head.loss_by_feat(cls_scores, bbox_preds,
                                            objectnesses, [gt_instances],
                                            img_metas)
        # When there is no truth, the cls loss should be nonzero but there
        # should be no box loss.
        empty_cls_loss = empty_gt_losses['loss_cls'].sum()
        empty_box_loss = empty_gt_losses['loss_bbox'].sum()
        empty_obj_loss = empty_gt_losses['loss_obj'].sum()
        self.assertEqual(
            empty_cls_loss.item(), 0,
            'there should be no cls loss when there are no true boxes')
        self.assertEqual(
            empty_box_loss.item(), 0,
            'there should be no box loss when there are no true boxes')
        self.assertGreater(empty_obj_loss.item(), 0,
                           'objectness loss should be non-zero')

        # When truth is non-empty then both cls and box loss should be nonzero
        # for random inputs
        head = YOLOXHead(
            num_classes=4,
            in_channels=1,
            stacked_convs=1,
            use_depthwise=True,
            train_cfg=train_cfg)
        assert isinstance(head.multi_level_cls_convs[0][0],
                          DepthwiseSeparableConvModule)
        head.use_l1 = True
        gt_instances = InstanceData(
            bboxes=torch.Tensor([[23.6667, 23.8757, 238.6326, 151.8874]]),
            labels=torch.LongTensor([2]))

        one_gt_losses = head.loss_by_feat(cls_scores, bbox_preds, objectnesses,
                                          [gt_instances], img_metas)
        onegt_cls_loss = one_gt_losses['loss_cls'].sum()
        onegt_box_loss = one_gt_losses['loss_bbox'].sum()
        onegt_obj_loss = one_gt_losses['loss_obj'].sum()
        onegt_l1_loss = one_gt_losses['loss_l1'].sum()
        self.assertGreater(onegt_cls_loss.item(), 0,
                           'cls loss should be non-zero')
        self.assertGreater(onegt_box_loss.item(), 0,
                           'box loss should be non-zero')
        self.assertGreater(onegt_obj_loss.item(), 0,
                           'obj loss should be non-zero')
        self.assertGreater(onegt_l1_loss.item(), 0,
                           'l1 loss should be non-zero')

        # Test groud truth out of bound
        gt_instances = InstanceData(
            bboxes=torch.Tensor([[s * 4, s * 4, s * 4 + 10, s * 4 + 10]]),
            labels=torch.LongTensor([2]))
        empty_gt_losses = head.loss_by_feat(cls_scores, bbox_preds,
                                            objectnesses, [gt_instances],
                                            img_metas)
        # When gt_bboxes out of bound, the assign results should be empty,
        # so the cls and bbox loss should be zero.
        empty_cls_loss = empty_gt_losses['loss_cls'].sum()
        empty_box_loss = empty_gt_losses['loss_bbox'].sum()
        empty_obj_loss = empty_gt_losses['loss_obj'].sum()
        self.assertEqual(
            empty_cls_loss.item(), 0,
            'there should be no cls loss when gt_bboxes out of bound')
        self.assertEqual(
            empty_box_loss.item(), 0,
            'there should be no box loss when gt_bboxes out of bound')
        self.assertGreater(empty_obj_loss.item(), 0,
                           'objectness loss should be non-zero')
