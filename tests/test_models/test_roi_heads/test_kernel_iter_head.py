import torch
from mmcv import ConfigDict

from mmdet.models.builder import build_head


def test_kernel_iter_head():
    num_stages = 3
    num_proposals = 10
    num_things_classes = 8
    num_stuff_classes = 5
    conv_kernel_size = 1
    channels = 4
    kernel_iter_head_cfg = ConfigDict(
        dict(
            type='KernelIterHead',
            num_stages=num_stages,
            stage_loss_weights=[1] * num_stages,
            proposal_feature_channel=channels,
            num_proposals=num_proposals,
            num_things_classes=num_things_classes,
            num_stuff_classes=num_stuff_classes,
            mask_assign_out_stride=4,
            mask_head=[
                dict(
                    type='KernelUpdateHead',
                    in_channels=channels,
                    out_channels=channels,
                    num_things_classes=num_things_classes,
                    num_stuff_classes=num_stuff_classes,
                    ignore_label=255,
                    num_cls_fcs=1,
                    num_mask_fcs=1,
                    act_cfg=dict(type='ReLU', inplace=True),
                    conv_kernel_size=conv_kernel_size,
                    feat_transform_cfg=dict(
                        conv_cfg=dict(type='Conv2d'), act_cfg=None),
                    mask_upsample_stride=2,
                    kernel_updator_cfg=dict(
                        type='KernelUpdator',
                        in_channels=channels,
                        feat_channels=channels,
                        out_channels=channels,
                        act_cfg=dict(type='ReLU', inplace=True),
                        norm_cfg=dict(type='LN')),
                    # attention + ffn + norm
                    attn_cfg=dict(
                        type='MultiheadAttention',
                        embed_dims=channels * conv_kernel_size**2,
                        num_heads=4,
                        attn_drop=0.0),
                    ffn_cfg=dict(
                        type='FFN',
                        embed_dims=channels,
                        feedforward_channels=channels * 8,
                        num_fcs=2,
                        act_cfg=dict(type='ReLU', inplace=True),
                        dropout=0.0),
                    attn_ffn_norm_cfg=dict(type='LN'),
                    loss_rank=dict(
                        type='CrossEntropyLoss',
                        use_sigmoid=False,
                        loss_weight=0.1),
                    loss_mask=dict(
                        type='CrossEntropyLoss',
                        use_sigmoid=True,
                        loss_weight=1.0),
                    loss_dice=dict(type='DiceLoss', loss_weight=4.0),
                    loss_cls=dict(
                        type='FocalLoss',
                        use_sigmoid=True,
                        gamma=2.0,
                        alpha=0.25,
                        loss_weight=2.0)) for _ in range(num_stages)
            ],
            train_cfg=[
                dict(
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
                    pos_weight=1) for _ in range(num_stages)
            ],
            test_cfg=None))

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

    # test for panoptic segmentation
    x = torch.rand((2, channels, 8, 8))
    proposal_feats = torch.rand((2, (num_proposals + num_stuff_classes),
                                 channels, conv_kernel_size, conv_kernel_size))
    mask_preds = torch.rand(2, (num_proposals + num_stuff_classes), 8, 8)

    head = build_head(kernel_iter_head_cfg)
    losses = head.forward_train(x, proposal_feats, mask_preds, img_metas,
                                gt_masks, gt_labels, gt_sem_seg, gt_sem_cls)
    assert isinstance(losses, dict)
    for i in range(num_stages):
        assert losses[f's{i}_loss_mask'].sum() > 0
        assert losses[f's{i}_loss_dice'].sum() > 0
        assert losses[f's{i}_loss_rank'].sum() > 0
        assert losses[f's{i}_loss_cls'].sum() > 0

    # test for instance segmentation
    num_stuff_classes = 0
    kernel_iter_head_cfg.num_stuff_classes = num_stuff_classes
    for i in range(num_stages):
        kernel_iter_head_cfg.mask_head[i].num_stuff_classes = num_stuff_classes

    x = torch.rand((2, channels, 8, 8))
    proposal_feats = torch.rand((2, (num_proposals + num_stuff_classes),
                                 channels, conv_kernel_size, conv_kernel_size))
    mask_preds = torch.rand(2, (num_proposals + num_stuff_classes), 8, 8)
    gt_sem_seg = [None, None]
    gt_sem_cls = [None, None]
    head = build_head(kernel_iter_head_cfg)
    losses = head.forward_train(x, proposal_feats, mask_preds, img_metas,
                                gt_masks, gt_labels, gt_sem_seg, gt_sem_cls)
    assert isinstance(losses, dict)
    for i in range(num_stages):
        assert losses[f's{i}_loss_mask'].sum() > 0
        assert losses[f's{i}_loss_dice'].sum() > 0
        assert losses[f's{i}_loss_cls'].sum() > 0
