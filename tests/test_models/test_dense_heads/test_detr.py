# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData

from mmdet.models.detectors.detr import DETR
from mmdet.structures import DetDataSample
from mmdet.utils import register_all_modules


class TestDETR(TestCase):

    def setUp(self) -> None:
        register_all_modules()

    def test_detr_head_loss(self):
        """Tests transformer head loss when truth is empty and non-empty."""
        s = 256
        metainfo = {
            'img_shape': (s, s),
            'scale_factor': (1, 1),
            'pad_shape': (s, s),
            'batch_input_shape': (s, s)
        }
        img_metas = DetDataSample()
        img_metas.set_metainfo(metainfo)

        batch_data_samples = []
        batch_data_samples.append(img_metas)
        config = ConfigDict(
            dict(
                num_query=100,
                backbone=dict(
                    type='ResNet',
                    depth=50,
                    num_stages=4,
                    out_indices=(3, ),
                    frozen_stages=1,
                    norm_cfg=dict(type='BN', requires_grad=False),
                    norm_eval=True,
                    style='pytorch',
                    init_cfg=dict(
                        type='Pretrained',
                        checkpoint='torchvision://resnet50')),
                neck=dict(
                    type='ChannelMapper',
                    in_channels=[2048],
                    kernel_size=1,
                    out_channels=256,
                    act_cfg=None,
                    norm_cfg=None,
                    num_outs=1),
                encoder=dict(  # DetrTransformerEncoder
                    num_layers=6,
                    layer_cfg=dict(  # DetrTransformerEncoderLayer
                        self_attn_cfg=dict(  # MultiheadAttention
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        ffn_cfg=dict(
                            embed_dims=256,
                            feedforward_channels=2048,
                            num_fcs=2,
                            ffn_drop=0.1,
                            act_cfg=dict(type='ReLU', inplace=True)))),
                decoder=dict(  # DetrTransformerDecoder
                    num_layers=6,
                    layer_cfg=dict(  # DetrTransformerDecoderLayer
                        self_attn_cfg=dict(  # MultiheadAttention
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        cross_attn_cfg=dict(  # MultiheadAttention
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        ffn_cfg=dict(
                            embed_dims=256,
                            feedforward_channels=2048,
                            num_fcs=2,
                            ffn_drop=0.1,
                            act_cfg=dict(type='ReLU', inplace=True))),
                    return_intermediate=True),
                positional_encoding_cfg=dict(num_feats=128, normalize=True),
                bbox_head=dict(
                    type='DETRHead',
                    num_classes=80,
                    embed_dims=256,
                    loss_cls=dict(
                        type='CrossEntropyLoss',
                        bg_cls_weight=0.1,
                        use_sigmoid=False,
                        loss_weight=1.0,
                        class_weight=1.0),
                    loss_bbox=dict(type='L1Loss', loss_weight=5.0),
                    loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
                # training and testing settings
                train_cfg=dict(
                    assigner=dict(
                        type='HungarianAssigner',
                        match_costs=[
                            dict(type='ClassificationCost', weight=1.),
                            dict(
                                type='BBoxL1Cost',
                                weight=5.0,
                                box_format='xywh'),
                            dict(type='IoUCost', iou_mode='giou', weight=2.0)
                        ])),
                test_cfg=dict(max_per_img=100)))

        model = DETR(**config)
        model.init_weights()
        random_image = torch.rand(1, 3, 256, 256)
        # Test that empty ground truth encourages the network to
        # predict background
        gt_instances = InstanceData()
        gt_instances.bboxes = torch.empty((0, 4))
        gt_instances.labels = torch.LongTensor([])
        img_metas.gt_instances = gt_instances
        batch_data_samples1 = []
        batch_data_samples1.append(img_metas)
        empty_gt_losses = model.loss(
            random_image, batch_data_samples=batch_data_samples1)
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
                    'there should be no iou loss when there are no true boxes')

        # When truth is non-empty then both cls and box loss should be nonzero
        # for random inputs
        gt_instances = InstanceData()
        gt_instances.bboxes = torch.Tensor(
            [[23.6667, 23.8757, 238.6326, 151.8874]])
        gt_instances.labels = torch.LongTensor([2])
        img_metas.gt_instances = gt_instances
        batch_data_samples2 = []
        batch_data_samples2.append(img_metas)
        one_gt_losses = model.loss(
            random_image, batch_data_samples=batch_data_samples2)
        for loss in one_gt_losses.values():
            self.assertGreater(
                loss.item(), 0,
                'cls loss, or box loss, or iou loss should be non-zero')

        # test loss and predict
        model._forward(random_image, batch_data_samples=batch_data_samples2)
        # test only predict
        model.predict(
            random_image, batch_data_samples=batch_data_samples2, rescale=True)
