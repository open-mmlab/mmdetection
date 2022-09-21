# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv import ConfigDict

from mmdet.models.dense_heads import DABDETRHead


def test_dab_detr_head_loss():
    """Tests dab-detr head loss when truth is empty and non-empty."""
    s = 256
    img_metas = [{
        'img_shape': (s, s, 3),
        'scale_factor': 1,
        'pad_shape': (s, s, 3),
        'batch_input_shape': (s, s)
    }]
    config = ConfigDict(
        dict(
            type='DABDETRHead',
            num_query=300,
            num_classes=91,
            in_channels=200,
            iter_update=True,
            random_refpoints_xy=False,
            transformer=dict(
                type='DabDetrTransformer',
                num_patterns=0,
                encoder=dict(
                    type='DabDetrTransformerEncoder',
                    num_layers=6,
                    transformerlayers=dict(
                        type='BaseTransformerLayer',
                        attn_cfgs=[
                            dict(
                                type='MultiheadAttention',
                                embed_dims=256,
                                num_heads=8,
                                dropout=0.1)
                        ],
                        feedforward_channels=2048,
                        ffn_dropout=0.1,
                        act_cfg=dict(type='PReLU'),
                        operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
                decoder=dict(
                    type='DabDetrTransformerDecoder',
                    query_dim=4,
                    num_layers=6,
                    query_scale_type='cond_elewise',
                    modulate_hw_attn=True,
                    bbox_embed_diff_each_layer=False,
                    transformerlayers=dict(
                        type='DabDetrTransformerDecoderLayer',
                        attn_cfgs=dict(
                            type='ConditionalAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        feedforward_channels=2048,
                        ffn_dropout=0.1,
                        sa_dropout=0.1,
                        ca_dropout=0.1,
                        keep_query_pos=False,
                        act_cfg=dict(type='PReLU'),
                        operation_order=('self_attn', 'norm', 'cross_attn',
                                         'norm', 'ffn', 'norm')),
                )),
            positional_encoding=dict(
                type='SinePositionalEncodingHW',
                num_feats=128,
                temperatureH=20,
                temperatureW=20,
                normalize=True),
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=5.0),
            loss_iou=dict(type='GIoULoss', loss_weight=2.0)))

    self = DABDETRHead(**config)
    self.init_weights()
    feat = [torch.rand(1, 200, 10, 10)]
    cls_scores, bbox_preds = self.forward(feat, img_metas)
    # Test that empty ground truth encourages the network to predict background
    gt_bboxes = [torch.empty((0, 4))]
    gt_labels = [torch.LongTensor([])]
    gt_bboxes_ignore = None
    empty_gt_losses = self.loss(cls_scores, bbox_preds, gt_bboxes, gt_labels,
                                img_metas, gt_bboxes_ignore)
    # When there is no truth, the cls loss should be nonzero but there should
    # be no box loss.
    for key, loss in empty_gt_losses.items():
        if 'cls' in key:
            assert loss.item() > 0, 'cls loss should be non-zero'
        elif 'bbox' in key:
            assert loss.item(
            ) == 0, 'there should be no box loss when there are no true boxes'
        elif 'iou' in key:
            assert loss.item(
            ) == 0, 'there should be no iou loss when there are no true boxes'

    # When truth is non-empty then both cls and box loss should be nonzero for
    # random inputs
    gt_bboxes = [
        torch.Tensor([[23.6667, 23.8757, 238.6326, 151.8874]]),
    ]
    gt_labels = [torch.LongTensor([2])]
    one_gt_losses = self.loss(cls_scores, bbox_preds, gt_bboxes, gt_labels,
                              img_metas, gt_bboxes_ignore)
    for loss in one_gt_losses.values():
        assert loss.item(
        ) > 0, 'cls loss, or box loss, or iou loss should be non-zero'

    # test forward_train
    self.forward_train(feat, img_metas, gt_bboxes, gt_labels)

    # test inference mode
    self.get_bboxes(cls_scores, bbox_preds, img_metas, rescale=True)
