_base_ = [
    '../_base_/datasets/coco_panoptic.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_1x.py'
]

num_stages = 3
conv_kernel_size = 1
num_proposals = 100
num_things_classes = 80
num_stuff_classes = 53

model = dict(
    type='KNet',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=0,
        add_extra_convs='on_input',
        num_outs=4),
    rpn_head=dict(
        type='KernelRPNHead',
        in_channels=256,
        out_channels=256,
        num_proposals=num_proposals,
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes,
        ignore_label=255,
        num_cls_fcs=1,
        num_loc_convs=1,
        num_seg_convs=1,
        localization_fpn_cfg=dict(
            type='SemanticFPN',
            in_channels=256,
            feat_channels=256,
            out_channels=256,
            start_level=0,
            end_level=3,
            output_level=1,
            positional_encoding_level=3,
            positional_encoding_cfg=dict(
                type='SinePositionalEncoding', num_feats=128, normalize=True),
            add_aux_conv=True,
            out_act_cfg=dict(type='ReLU'),
            conv_cfg=None,
            norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)),
        conv_kernel_size=conv_kernel_size,
        norm_cfg=dict(type='GN', num_groups=32),
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
            loss_weight=1.0)),
    roi_head=dict(
        type='KernelIterHead',
        num_stages=num_stages,
        stage_loss_weights=[1] * num_stages,
        proposal_feature_channel=256,
        num_proposals=num_proposals,
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes,
        mask_assign_out_stride=4,
        mask_head=[
            dict(
                type='KernelUpdateHead',
                in_channels=256,
                out_channels=256,
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
                    in_channels=256,
                    feat_channels=256,
                    out_channels=256,
                    act_cfg=dict(type='ReLU', inplace=True),
                    norm_cfg=dict(type='LN')),
                # attention + ffn + norm
                attn_cfg=dict(
                    type='MultiheadAttention',
                    embed_dims=256 * conv_kernel_size**2,
                    num_heads=8,
                    attn_drop=0.0),
                ffn_cfg=dict(
                    type='FFN',
                    embed_dims=256,
                    feedforward_channels=2048,
                    num_fcs=2,
                    act_cfg=dict(type='ReLU', inplace=True),
                    dropout=0.0),
                attn_ffn_norm_cfg=dict(type='LN'),
                loss_rank=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=0.1),
                loss_mask=dict(
                    type='CrossEntropyLoss', use_sigmoid=True,
                    loss_weight=1.0),
                loss_dice=dict(type='DiceLoss', loss_weight=4.0),
                loss_cls=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=2.0)) for _ in range(num_stages)
        ]),
    panoptic_fusion_head=dict(
        type='KNetFusionHead',
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes,
        num_proposals=num_proposals),
    # training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaskHungarianAssigner',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                dice_cost=dict(
                    type='DiceCost',
                    weight=4.0,
                    pred_act=True,
                    naive_dice=False),
                mask_cost=dict(type='MaskCost', weight=1.0, use_sigmoid=True)),
            sampler=dict(type='MaskPseudoSampler'),
            pos_weight=1),
        rcnn=[
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
        ]),
    test_cfg=dict(
        rpn=None,
        rcnn=None,
        fusion=dict(
            # instance segmentation and panoptic segmentation
            # can't be turn on at the same time
            panoptic_on=True,
            instance_on=False,
            # test cfg for panoptic segmentation
            overlap_thr=0.6,
            instance_score_thr=0.3,
            # tes cfg for instance segmentation
            max_per_img=num_proposals,
            mask_thr=0.5)))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadPanopticAnnotations',
        with_bbox=True,
        with_mask=True,
        with_seg=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks', 'gt_semantic_seg']),
]
data = dict(train=dict(pipeline=train_pipeline))

# optimizer
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.0001,
    weight_decay=0.05,
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.25)}))
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=1, norm_type=2))

# learning policy
lr_config = dict(warmup_iters=1000)
