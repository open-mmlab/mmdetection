import numpy as np
import torch
from mmcv import ConfigDict

from mmdet.core.mask import BitmapMasks
from mmdet.models.dense_heads import MaskFormerHead


def test_maskformer_head_loss():
    """Tests head loss when truth is empty and non-empty."""
    img_metas = [{
        'batch_input_shape': (960, 1280),
        'img_shape': (958, 1271, 3),
        'ori_shape': (479, 635, 3)
    }, {
        'batch_input_shape': (960, 1280),
        'img_shape': (954, 1280, 3),
        'ori_shape': (477, 640, 3)
    }]
    feats = [
        torch.rand((2, 256 * 2**i, 30 * 2**(3 - i), 40 * 2**(3 - i)))
        for i in range(4)
    ]

    config = ConfigDict(
        dict(
            type='MaskFormerHead',
            in_channels=[256, 512, 1024, 2048],  # pass to pixel_decoder inside
            feat_channels=256,
            out_channels=256,
            num_things_classes=80,
            num_stuff_classes=53,
            num_queries=100,
            pixel_decoder=dict(
                type='TransformerEncoderPixelDecoder',
                norm_cfg=dict(type='GN', num_groups=32),
                act_cfg=dict(type='ReLU'),
                encoder=dict(
                    type='DetrTransformerEncoder',
                    num_layers=6,
                    transformerlayers=dict(
                        type='BaseTransformerLayer',
                        attn_cfgs=dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            attn_drop=0.1,
                            proj_drop=0.1,
                            dropout_layer=None,
                            batch_first=False),
                        ffn_cfgs=dict(
                            embed_dims=256,
                            feedforward_channels=2048,
                            num_fcs=2,
                            act_cfg=dict(type='ReLU', inplace=True),
                            ffn_drop=0.1,
                            dropout_layer=None,
                            add_identity=True),
                        operation_order=('self_attn', 'norm', 'ffn', 'norm'),
                        norm_cfg=dict(type='LN'),
                        init_cfg=None,
                        batch_first=False),
                    init_cfg=None),
                positional_encoding=dict(
                    type='SinePositionalEncoding',
                    num_feats=128,
                    normalize=True)),
            enforce_decoder_input_project=False,
            positional_encoding=dict(
                type='SinePositionalEncoding', num_feats=128, normalize=True),
            transformer_decoder=dict(
                type='DetrTransformerDecoder',
                return_intermediate=True,
                num_layers=6,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=dict(
                        type='MultiheadAttention',
                        embed_dims=256,
                        num_heads=8,
                        attn_drop=0.1,
                        proj_drop=0.1,
                        dropout_layer=None,
                        batch_first=False),
                    ffn_cfgs=dict(
                        embed_dims=256,
                        feedforward_channels=2048,
                        num_fcs=2,
                        act_cfg=dict(type='ReLU', inplace=True),
                        ffn_drop=0.1,
                        dropout_layer=None,
                        add_identity=True),
                    # the following parameter was not used,
                    # just make current api happy
                    feedforward_channels=2048,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')),
                init_cfg=None),
            loss_cls=dict(
                type='CrossEntropyLoss',
                bg_cls_weight=0.1,
                use_sigmoid=False,
                loss_weight=1.0,
                reduction='mean',
                class_weight=1.0),
            loss_mask=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                reduction='mean',
                loss_weight=20.0),
            loss_dice=dict(
                type='DiceLoss',
                use_sigmoid=True,
                activate=True,
                reduction='mean',
                naive_dice=True,
                eps=1.0,
                loss_weight=1.0),
            train_cfg=dict(
                assigner=dict(
                    type='MaskHungarianAssigner',
                    cls_cost=dict(type='ClassificationCost', weight=1.0),
                    mask_cost=dict(type='MaskFocalLossCost', weight=20.0),
                    dice_cost=dict(
                        type='DiceCost', weight=1.0, pred_act=True, eps=1.0)),
                sampler=dict(type='MaskPseudoSampler')),
            test_cfg=dict(object_mask_thr=0.8, iou_thr=0.8)))
    self = MaskFormerHead(**config)
    self.init_weights()
    all_cls_scores, all_mask_preds = self.forward(feats, img_metas)
    # Test that empty ground truth encourages the network to predict background
    gt_labels_list = [torch.LongTensor([]), torch.LongTensor([])]
    gt_masks_list = [
        torch.zeros((0, 960, 1280)).long(),
        torch.zeros((0, 960, 1280)).long()
    ]

    empty_gt_losses = self.loss(all_cls_scores, all_mask_preds, gt_labels_list,
                                gt_masks_list, img_metas)
    # When there is no truth, the cls loss should be nonzero but there should
    # be no mask loss.
    for key, loss in empty_gt_losses.items():
        if 'cls' in key:
            assert loss.item() > 0, 'cls loss should be non-zero'
        elif 'mask' in key:
            assert loss.item(
            ) == 0, 'there should be no mask loss when there are no true mask'
        elif 'dice' in key:
            assert loss.item(
            ) == 0, 'there should be no dice loss when there are no true mask'

    # when truth is non-empty then both cls, mask, dice loss should be nonzero
    # random inputs
    gt_labels_list = [
        torch.tensor([10, 100]).long(),
        torch.tensor([100, 10]).long()
    ]
    mask1 = torch.zeros((2, 960, 1280)).long()
    mask1[0, :500] = 1
    mask1[1, 500:] = 1
    mask2 = torch.zeros((2, 960, 1280)).long()
    mask2[0, :, :500] = 1
    mask2[1, :, 500:] = 1
    gt_masks_list = [mask1, mask2]
    two_gt_losses = self.loss(all_cls_scores, all_mask_preds, gt_labels_list,
                              gt_masks_list, img_metas)
    for loss in two_gt_losses.values():
        assert loss.item() > 0, 'all loss should be non-zero'

    # test forward_train
    gt_bboxes = None
    gt_labels = [
        torch.tensor([10]).long(),
        torch.tensor([10]).long(),
    ]
    thing_mask1 = np.zeros((1, 960, 1280), dtype=np.int32)
    thing_mask1[0, :500] = 1
    thing_mask2 = np.zeros((1, 960, 1280), dtype=np.int32)
    thing_mask2[0, :, 500:] = 1
    gt_masks = [
        BitmapMasks(thing_mask1, 960, 1280),
        BitmapMasks(thing_mask2, 960, 1280),
    ]
    stuff_mask1 = torch.zeros((1, 960, 1280)).long()
    stuff_mask1[0, :500] = 10
    stuff_mask1[0, 500:] = 100
    stuff_mask2 = torch.zeros((1, 960, 1280)).long()
    stuff_mask2[0, :, 500:] = 10
    stuff_mask2[0, :, :500] = 100
    gt_semantic_seg = [stuff_mask1, stuff_mask2]

    self.forward_train(feats, img_metas, gt_bboxes, gt_labels, gt_masks,
                       gt_semantic_seg)

    # test inference mode
    self.simple_test(feats, img_metas)
