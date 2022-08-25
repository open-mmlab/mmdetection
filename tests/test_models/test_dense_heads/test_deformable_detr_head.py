# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData

from mmdet.models.dense_heads import DeformableDETRHead
from mmdet.utils import register_all_modules


class TestDeformableDETRHead(TestCase):

    def setUp(self):
        register_all_modules()

    def test_detr_head_loss(self):
        """Tests transformer head loss when truth is empty and non-empty."""
        s = 256
        img_metas = [{
            'img_shape': (s, s),
            'scale_factor': (1, 1),
            'pad_shape': (s, s),
            'batch_input_shape': (s, s)
        }]
        config = ConfigDict(
            dict(
                num_classes=4,
                in_channels=2048,
                sync_cls_avg_factor=True,
                as_two_stage=False,
                transformer=dict(
                    type='DeformableDetrTransformer',
                    encoder=dict(
                        type='DetrTransformerEncoder',
                        num_layers=6,
                        transformerlayers=dict(
                            type='BaseTransformerLayer',
                            attn_cfgs=dict(
                                type='MultiScaleDeformableAttention',
                                embed_dims=256),
                            feedforward_channels=1024,
                            ffn_dropout=0.1,
                            operation_order=('self_attn', 'norm', 'ffn',
                                             'norm'))),
                    decoder=dict(
                        type='DeformableDetrTransformerDecoder',
                        num_layers=6,
                        return_intermediate=True,
                        transformerlayers=dict(
                            type='DetrTransformerDecoderLayer',
                            attn_cfgs=[
                                dict(
                                    type='MultiheadAttention',
                                    embed_dims=256,
                                    num_heads=8,
                                    dropout=0.1),
                                dict(
                                    type='MultiScaleDeformableAttention',
                                    embed_dims=256)
                            ],
                            feedforward_channels=1024,
                            ffn_dropout=0.1,
                            operation_order=('self_attn', 'norm', 'cross_attn',
                                             'norm', 'ffn', 'norm')))),
                positional_encoding=dict(
                    type='SinePositionalEncoding',
                    num_feats=128,
                    normalize=True,
                    offset=-0.5),
                loss_cls=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=2.0),
                loss_bbox=dict(type='L1Loss', loss_weight=5.0),
                loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
            train_cfg=dict(
                assigner=dict(
                    type='HungarianAssigner',
                    match_costs=[
                        dict(type='FocalLossCost', weight=2.0),
                        dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                        dict(type='IoUCost', iou_mode='giou', weight=2.0)
                    ])),
            test_cfg=dict(max_per_img=100))

        deformable_detr_head = DeformableDETRHead(**config)
        deformable_detr_head.init_weights()
        feat = [
            torch.rand(1, 256, s // stride, s // stride)
            for stride in [8, 16, 32, 64]
        ]
        outs = deformable_detr_head.forward(feat, img_metas)
        # Test that empty ground truth encourages the network to
        # predict background
        gt_instances = InstanceData()
        gt_instances.bboxes = torch.empty((0, 4))
        gt_instances.labels = torch.LongTensor([])
        empty_gt_losses = deformable_detr_head.loss_by_feat(
            *outs, [gt_instances], img_metas)
        # When there is no truth, the cls loss should be nonzero but there
        # should be no box loss.
        for key, loss in empty_gt_losses.items():
            if 'cls' in key:
                self.assertGreater(loss.item(), 0,
                                   'cls loss should be non-zero')
            elif 'bbox' in key:
                self.assertEqual(
                    loss.item(), 0,
                    'there should be no box loss when no ground true boxes')
            elif 'iou' in key:
                self.assertEqual(
                    loss.item(), 0,
                    'there should be no iou loss when no ground true boxes')

        # When truth is non-empty then both cls and box loss should be nonzero
        # for random inputs
        gt_instances = InstanceData()
        gt_instances.bboxes = torch.Tensor(
            [[23.6667, 23.8757, 238.6326, 151.8874]])
        gt_instances.labels = torch.LongTensor([2])
        one_gt_losses = deformable_detr_head.loss_by_feat(
            *outs, [gt_instances], img_metas)
        for loss in one_gt_losses.values():
            self.assertGreater(
                loss.item(), 0,
                'cls loss, or box loss, or iou loss should be non-zero')

        # test predict_by_feat
        deformable_detr_head.predict_by_feat(*outs, img_metas, rescale=False)
