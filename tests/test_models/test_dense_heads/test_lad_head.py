# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import numpy as np
import torch
from mmengine import Config
from mmengine.structures import InstanceData

from mmdet import *  # noqa
from mmdet.models.dense_heads import LADHead, lad_head
from mmdet.models.dense_heads.lad_head import levels_to_images


class TestLADHead(TestCase):

    def test_lad_head_loss(self):
        """Tests lad head loss when truth is empty and non-empty."""

        class mock_skm:

            def GaussianMixture(self, *args, **kwargs):
                return self

            def fit(self, loss):
                pass

            def predict(self, loss):
                components = np.zeros_like(loss, dtype=np.long)
                return components.reshape(-1)

            def score_samples(self, loss):
                scores = np.random.random(len(loss))
                return scores

        lad_head.skm = mock_skm()

        s = 256
        img_metas = [{
            'img_shape': (s, s, 3),
            'pad_shape': (s, s, 3),
            'scale_factor': 1
        }]
        train_cfg = Config(
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.1,
                    neg_iou_thr=0.1,
                    min_pos_iou=0,
                    ignore_iof_thr=-1),
                allowed_border=-1,
                pos_weight=-1,
                debug=False))
        # since Focal Loss is not supported on CPU
        # since Focal Loss is not supported on CPU
        lad = LADHead(
            num_classes=4,
            in_channels=1,
            train_cfg=train_cfg,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
            loss_bbox=dict(type='GIoULoss', loss_weight=1.3),
            loss_centerness=dict(
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=0.5))
        teacher_model = LADHead(
            num_classes=4,
            in_channels=1,
            train_cfg=train_cfg,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
            loss_bbox=dict(type='GIoULoss', loss_weight=1.3),
            loss_centerness=dict(
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=0.5))
        feat = [
            torch.rand(1, 1, s // feat_size, s // feat_size)
            for feat_size in [4, 8, 16, 32, 64]
        ]
        lad.init_weights()
        teacher_model.init_weights()

        # Test that empty ground truth encourages the network to predict
        # background
        gt_instances = InstanceData()
        gt_instances.bboxes = torch.empty((0, 4))
        gt_instances.labels = torch.LongTensor([])
        batch_gt_instances_ignore = None

        outs_teacher = teacher_model(feat)
        label_assignment_results = teacher_model.get_label_assignment(
            *outs_teacher, [gt_instances], img_metas,
            batch_gt_instances_ignore)

        outs = teacher_model(feat)
        empty_gt_losses = lad.loss_by_feat(*outs, [gt_instances], img_metas,
                                           batch_gt_instances_ignore,
                                           label_assignment_results)
        # When there is no truth, the cls loss should be nonzero but there
        # should be no box loss.
        empty_cls_loss = empty_gt_losses['loss_cls']
        empty_box_loss = empty_gt_losses['loss_bbox']
        empty_iou_loss = empty_gt_losses['loss_iou']
        self.assertGreater(empty_cls_loss.item(), 0,
                           'cls loss should be non-zero')
        self.assertEqual(
            empty_box_loss.item(), 0,
            'there should be no box loss when there are no true boxes')
        self.assertEqual(
            empty_iou_loss.item(), 0,
            'there should be no box loss when there are no true boxes')

        # When truth is non-empty then both cls and box loss should be nonzero
        # for random inputs
        gt_instances = InstanceData()
        gt_instances.bboxes = torch.Tensor(
            [[23.6667, 23.8757, 238.6326, 151.8874]])
        gt_instances.labels = torch.LongTensor([2])
        batch_gt_instances_ignore = None

        label_assignment_results = teacher_model.get_label_assignment(
            *outs_teacher, [gt_instances], img_metas,
            batch_gt_instances_ignore)
        one_gt_losses = lad.loss_by_feat(*outs, [gt_instances], img_metas,
                                         batch_gt_instances_ignore,
                                         label_assignment_results)
        onegt_cls_loss = one_gt_losses['loss_cls']
        onegt_box_loss = one_gt_losses['loss_bbox']
        onegt_iou_loss = one_gt_losses['loss_iou']
        self.assertGreater(onegt_cls_loss.item(), 0,
                           'cls loss should be non-zero')
        self.assertGreater(onegt_box_loss.item(), 0,
                           'box loss should be non-zero')
        self.assertGreater(onegt_iou_loss.item(), 0,
                           'box loss should be non-zero')
        n, c, h, w = 10, 4, 20, 20
        mlvl_tensor = [torch.ones(n, c, h, w) for i in range(5)]
        results = levels_to_images(mlvl_tensor)
        self.assertEqual(len(results), n)
        self.assertEqual(results[0].size(), (h * w * 5, c))
        self.assertTrue(lad.with_score_voting)

        lad = LADHead(
            num_classes=4,
            in_channels=1,
            train_cfg=train_cfg,
            anchor_generator=dict(
                type='AnchorGenerator',
                ratios=[1.0],
                octave_base_scale=8,
                scales_per_octave=1,
                strides=[8]),
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
            loss_bbox=dict(type='GIoULoss', loss_weight=1.3),
            loss_centerness=dict(
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=0.5))
        cls_scores = [torch.ones(2, 4, 5, 5)]
        bbox_preds = [torch.ones(2, 4, 5, 5)]
        iou_preds = [torch.ones(2, 1, 5, 5)]
        cfg = Config(
            dict(
                nms_pre=1000,
                min_bbox_size=0,
                score_thr=0.05,
                nms=dict(type='nms', iou_threshold=0.6),
                max_per_img=100))
        rescale = False
        lad.predict_by_feat(
            cls_scores, bbox_preds, iou_preds, img_metas, cfg, rescale=rescale)
