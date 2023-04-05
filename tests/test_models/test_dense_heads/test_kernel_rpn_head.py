import torch
from mmcv import ConfigDict

from mmdet.models.builder import build_head


def test_kernel_rpn_head():
    # test for panoptic segmentation
    channels = 4
    conv_kernel_size = 1
    num_things_classes = 8
    num_stuff_classes = 5
    num_proposals = 10
    kernel_rpn_head_cfg = ConfigDict(
        dict(
            type='KernelRPNHead',
            in_channels=channels,
            out_channels=channels,
            num_proposals=num_proposals,
            num_things_classes=num_things_classes,
            num_stuff_classes=num_stuff_classes,
            ignore_label=255,
            num_cls_fcs=1,
            num_loc_convs=1,
            num_seg_convs=1,
            localization_fpn_cfg=dict(
                type='SemanticFPN',
                in_channels=channels,
                feat_channels=channels,
                out_channels=channels,
                start_level=0,
                end_level=3,
                output_level=1,
                positional_encoding_level=3,
                positional_encoding_cfg=dict(
                    type='SinePositionalEncoding',
                    num_feats=channels // 2,
                    normalize=True),
                add_aux_conv=True,
                out_act_cfg=dict(type='ReLU'),
                conv_cfg=None,
                norm_cfg=dict(
                    type='GN', num_groups=channels // 2, requires_grad=True)),
            conv_kernel_size=conv_kernel_size,
            norm_cfg=dict(type='GN', num_groups=channels // 2),
            feat_scale_factor=2,
            loss_rank=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.1),
            loss_mask=dict(
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
            loss_dice=dict(type='DiceLoss', loss_weight=4.0),
            loss_seg=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0),
            train_cfg=dict(
                assigner=dict(
                    type='MaskHungarianAssigner',
                    cls_cost=dict(type='FocalLossCost', weight=2.0),
                    dice_cost=dict(
                        type='DiceCost',
                        weight=4.0,
                        pred_act=True,
                        naive_dice=False),
                    mask_cost=dict(
                        type='MaskCost', weight=1.0, use_sigmoid=True)),
                sampler=dict(type='MaskPseudoSampler'),
                pos_weight=1),
            test_cfg=None))

    x = [
        torch.rand((2, channels, 2**(4 - i), 2**(4 - i))) for i in range(0, 4)
    ]
    img_metas = [{
        'batch_input_shape': (64, 64),
        'pad_shape': (64, 60, 3),
        'img_shape': (63, 59, 3),
        'ori_shape': (63, 59, 3)
    }, {
        'batch_input_shape': (64, 64),
        'pad_shape': (60, 64, 3),
        'img_shape': (59, 63, 3),
        'ori_shape': (59, 63, 3)
    }]
    ins_mask1 = torch.zeros((1, 16, 16)).float()
    ins_mask1[:, :8] = 1
    stuff_mask1 = torch.zeros((1, 16, 16)).float()
    stuff_mask1[:, 8:] = 1
    ins_mask2 = torch.zeros((1, 16, 16)).float()
    ins_mask2[:, 8:] = 1
    stuff_mask2 = torch.zeros((1, 16, 16)).float()
    stuff_mask2[:, :8] = 1
    gt_masks = [ins_mask1, ins_mask2]
    gt_sem_seg = [stuff_mask1, stuff_mask2]
    gt_labels = [
        torch.tensor([
            2,
        ], dtype=torch.long),
        torch.tensor([
            4,
        ], dtype=torch.long)
    ]
    gt_sem_cls = [
        torch.tensor([
            10,
        ], dtype=torch.long),
        torch.tensor([
            11,
        ], dtype=torch.long)
    ]
    head = build_head(kernel_rpn_head_cfg)
    out = head.forward_train(x, img_metas, gt_masks, gt_labels, gt_sem_seg,
                             gt_sem_cls)
    assert len(out) == 4
    assert isinstance(out[0], dict)
    assert out[0]['loss_rpn_mask'].sum() > 0
    assert out[0]['loss_rpn_dice'].sum() > 0
    assert out[0]['loss_rpn_rank'].sum() > 0
    assert out[0]['loss_rpn_seg'].sum() > 0

    assert out[1].shape == (2, channels, 8, 8)
    assert out[2].shape == (2, (num_proposals + num_stuff_classes), channels,
                            conv_kernel_size, conv_kernel_size)
    assert out[3].shape == (2, (num_proposals + num_stuff_classes), 8, 8)

    out = head.simple_test_rpn(x, img_metas)
    assert len(out) == 3
    assert out[0].shape == (2, channels, 8, 8)
    assert out[1].shape == (2, (num_proposals + num_stuff_classes), channels,
                            conv_kernel_size, conv_kernel_size)
    assert out[2].shape == (2, (num_proposals + num_stuff_classes), 8, 8)

    # test for instance segmentation
    kernel_rpn_head_cfg.update(num_stuff_classes=0)
    gt_sem_seg = [None, None]
    gt_sem_cls = [None, None]
    head = build_head(kernel_rpn_head_cfg)
    out = head.forward_train(x, img_metas, gt_masks, gt_labels, gt_sem_seg,
                             gt_sem_cls)
    assert len(out) == 4
    assert isinstance(out[0], dict)
    assert out[0]['loss_rpn_mask'].sum() > 0
    assert out[0]['loss_rpn_dice'].sum() > 0
    assert out[0]['loss_rpn_rank'].sum() > 0
    assert out[1].shape == (2, channels, 8, 8)
    assert out[2].shape == (2, num_proposals, channels, conv_kernel_size,
                            conv_kernel_size)
    assert out[3].shape == (2, num_proposals, 8, 8)

    out = head.simple_test_rpn(x, img_metas)
    assert len(out) == 3
    assert out[0].shape == (2, channels, 8, 8)
    assert out[1].shape == (2, num_proposals, channels, conv_kernel_size,
                            conv_kernel_size)
    assert out[2].shape == (2, num_proposals, 8, 8)
