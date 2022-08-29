# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv import Config

from mmdet.models.dense_heads import HDeformableDETRHead


def test_detr_head_loss():
    """Tests transformer head loss when truth is empty and non-empty."""
    s = 256
    img_metas = [{
        'img_shape': (s, s, 3),
        'scale_factor': 1,
        'pad_shape': (s, s, 3),
        'batch_input_shape': (s, s)
    }]
    config = Config(
        dict(
            type='HDeformableDETRHead',
            num_queries_one2one=300,
            num_queries_one2many=1500,
            k_one2many=6,
            lambda_one2many=1.0,
            num_classes=80,
            in_channels=256,
            sync_cls_avg_factor=True,
            as_two_stage=True,
            with_box_refine=True,
            mixed_selection=True,
            transformer=dict(
                type='DeformableDetrTransformer',
                encoder=dict(
                    type='DetrTransformerEncoder',
                    num_layers=6,
                    transformerlayers=dict(
                        type='BaseTransformerLayer',
                        attn_cfgs=dict(
                            type='MultiScaleDeformableAttention',
                            embed_dims=256,
                            dropout=0.0),
                        feedforward_channels=2048,
                        ffn_dropout=0.0,
                        operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
                decoder=dict(
                    type='DeformableDetrTransformerDecoder',
                    num_layers=6,
                    return_intermediate=True,
                    look_forward_twice=True,
                    transformerlayers=dict(
                        type='DetrTransformerDecoderLayer',
                        attn_cfgs=[
                            dict(
                                type='MultiheadAttention',
                                embed_dims=256,
                                num_heads=8,
                                dropout=0.0),
                            dict(
                                type='MultiScaleDeformableAttention',
                                embed_dims=256,
                                dropout=0.0)
                        ],
                        feedforward_channels=2048,
                        ffn_dropout=0.0,
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
            loss_iou=dict(type='GIoULoss', loss_weight=2.0),
            train_cfg=dict(
                assigner=dict(
                    type='HungarianAssigner',
                    cls_cost=dict(type='FocalLossCost', weight=2.0),
                    reg_cost=dict(
                        type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                    iou_cost=dict(type='IoUCost', iou_mode='giou',
                                  weight=2.0)))))

    self = HDeformableDETRHead(**config)
    self.init_weights()
    feat = [
        torch.rand(1, 256, 64, 64),
        torch.rand(1, 256, 32, 32),
        torch.rand(1, 256, 16, 16),
        torch.rand(1, 256, 8, 8)
    ]
    (cls_scores_one2one, bbox_preds_one2one, cls_scores_one2many,
     bbox_preds_one2many, enc_class,
     enc_coord) = self.forward(feat, img_metas)
    # Test that empty ground truth encourages the network to predict background
    gt_bboxes = [torch.empty((0, 4))]
    gt_labels = [torch.LongTensor([])]
    gt_bboxes_ignore = None
    empty_gt_losses = self.loss(cls_scores_one2one, bbox_preds_one2one,
                                cls_scores_one2many, bbox_preds_one2many,
                                enc_class, enc_coord, gt_bboxes, gt_labels,
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
    one_gt_losses = self.loss(cls_scores_one2one, bbox_preds_one2one,
                              cls_scores_one2many, bbox_preds_one2many,
                              enc_class, enc_coord, gt_bboxes, gt_labels,
                              img_metas, gt_bboxes_ignore)
    for loss in one_gt_losses.values():
        assert loss.item(
        ) > 0, 'cls loss, or box loss, or iou loss should be non-zero'

    # test forward_train
    self.forward_train(feat, img_metas, gt_bboxes, gt_labels)

    # test inference mode
    self.get_bboxes(
        cls_scores_one2one,
        bbox_preds_one2one,
        cls_scores_one2many,
        bbox_preds_one2many,
        enc_class,
        enc_coord,
        img_metas,
        rescale=True)
