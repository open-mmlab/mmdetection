# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
from mmengine.config import Config
from mmengine.structures import InstanceData

from mmdet.models.dense_heads import YOLOV3Head


class TestYOLOV3Head(TestCase):

    def test_yolo_head_loss(self):
        """Tests YOLO head loss when truth is empty and non-empty."""
        s = 256
        img_metas = [{
            'img_shape': (s, s, 3),
            'scale_factor': 1,
        }]
        head = YOLOV3Head(
            num_classes=4,
            in_channels=[1, 1, 1],
            out_channels=[1, 1, 1],
            train_cfg=Config(
                dict(
                    assigner=dict(
                        type='GridAssigner',
                        pos_iou_thr=0.5,
                        neg_iou_thr=0.5,
                        min_pos_iou=0))))
        head.init_weights()

        # YOLO head expects a multiple levels of features per image
        feats = [
            torch.rand(1, 1, s // stride[1], s // stride[0])
            for stride in head.prior_generator.strides
        ]
        predmaps, = head.forward(feats)

        # Test that empty ground truth encourages the network to
        # predict background
        gt_instances = InstanceData()
        gt_instances.bboxes = torch.empty((0, 4))
        gt_instances.labels = torch.LongTensor([])

        empty_gt_losses = head.loss_by_feat(predmaps, [gt_instances],
                                            img_metas)
        # When there is no truth, the conf loss should be nonzero but
        # cls loss and xy&wh loss should be zero
        empty_cls_loss = sum(empty_gt_losses['loss_cls']).item()
        empty_conf_loss = sum(empty_gt_losses['loss_conf']).item()
        empty_xy_loss = sum(empty_gt_losses['loss_xy']).item()
        empty_wh_loss = sum(empty_gt_losses['loss_wh']).item()
        self.assertGreater(empty_conf_loss, 0, 'conf loss should be non-zero')
        self.assertEqual(
            empty_cls_loss, 0,
            'there should be no cls loss when there are no true boxes')
        self.assertEqual(
            empty_xy_loss, 0,
            'there should be no xy loss when there are no true boxes')
        self.assertEqual(
            empty_wh_loss, 0,
            'there should be no wh loss when there are no true boxes')

        # When truth is non-empty then all conf, cls loss and xywh loss
        # should be nonzero for random inputs
        gt_instances = InstanceData()
        gt_instances.bboxes = torch.Tensor(
            [[23.6667, 23.8757, 238.6326, 151.8874]])
        gt_instances.labels = torch.LongTensor([2])

        one_gt_losses = head.loss_by_feat(predmaps, [gt_instances], img_metas)
        one_gt_cls_loss = sum(one_gt_losses['loss_cls']).item()
        one_gt_conf_loss = sum(one_gt_losses['loss_conf']).item()
        one_gt_xy_loss = sum(one_gt_losses['loss_xy']).item()
        one_gt_wh_loss = sum(one_gt_losses['loss_wh']).item()
        self.assertGreater(one_gt_conf_loss, 0, 'conf loss should be non-zero')
        self.assertGreater(one_gt_cls_loss, 0, 'cls loss should be non-zero')
        self.assertGreater(one_gt_xy_loss, 0, 'xy loss should be non-zero')
        self.assertGreater(one_gt_wh_loss, 0, 'wh loss should be non-zero')
