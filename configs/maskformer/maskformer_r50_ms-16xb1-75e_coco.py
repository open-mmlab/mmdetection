_base_ = [
    '../_base_/datasets/coco_panoptic.py', '../_base_/default_runtime.py'
]

data_preprocessor = dict(
    type='DetDataPreprocessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_size_divisor=1,
    pad_mask=True,
    mask_pad_value=0,
    pad_seg=True,
    seg_pad_value=255)

num_things_classes = 80
num_stuff_classes = 53
num_classes = num_things_classes + num_stuff_classes
model = dict(
    type='MaskFormer',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    panoptic_head=dict(
        type='MaskFormerHead',
        in_channels=[256, 512, 1024, 2048],  # pass to pixel_decoder inside
        feat_channels=256,
        out_channels=256,
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes,
        num_queries=100,
        pixel_decoder=dict(
            type='TransformerEncoderPixelDecoder',
            norm_cfg=dict(type='GN', num_groups=32),
            act_cfg=dict(type='ReLU'),
            encoder=dict(  # DetrTransformerEncoder
                num_layers=6,
                layer_cfg=dict(  # DetrTransformerEncoderLayer
                    self_attn_cfg=dict(  # MultiheadAttention
                        embed_dims=256,
                        num_heads=8,
                        dropout=0.1,
                        batch_first=True),
                    ffn_cfg=dict(
                        embed_dims=256,
                        feedforward_channels=2048,
                        num_fcs=2,
                        ffn_drop=0.1,
                        act_cfg=dict(type='ReLU', inplace=True)))),
            positional_encoding=dict(num_feats=128, normalize=True)),
        enforce_decoder_input_project=False,
        positional_encoding=dict(num_feats=128, normalize=True),
        transformer_decoder=dict(  # DetrTransformerDecoder
            num_layers=6,
            layer_cfg=dict(  # DetrTransformerDecoderLayer
                self_attn_cfg=dict(  # MultiheadAttention
                    embed_dims=256,
                    num_heads=8,
                    dropout=0.1,
                    batch_first=True),
                cross_attn_cfg=dict(  # MultiheadAttention
                    embed_dims=256,
                    num_heads=8,
                    dropout=0.1,
                    batch_first=True),
                ffn_cfg=dict(
                    embed_dims=256,
                    feedforward_channels=2048,
                    num_fcs=2,
                    ffn_drop=0.1,
                    act_cfg=dict(type='ReLU', inplace=True))),
            return_intermediate=True),
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0,
            reduction='mean',
            class_weight=[1.0] * num_classes + [0.1]),
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
            loss_weight=1.0)),
    panoptic_fusion_head=dict(
        type='MaskFormerFusionHead',
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes,
        loss_panoptic=None,
        init_cfg=None),
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='ClassificationCost', weight=1.0),
                dict(type='FocalLossCost', weight=20.0, binary_input=True),
                dict(type='DiceCost', weight=1.0, pred_act=True, eps=1.0)
            ]),
        sampler=dict(type='MaskPseudoSampler')),
    test_cfg=dict(
        panoptic_on=True,
        # For now, the dataset does not support
        # evaluating semantic segmentation metric.
        semantic_on=False,
        instance_on=False,
        # max_per_image is for instance segmentation.
        max_per_image=100,
        object_mask_thr=0.8,
        iou_thr=0.8,
        # In MaskFormer's panoptic postprocessing,
        # it will not filter masks whose score is smaller than 0.5 .
        filter_low_score=False),
    init_cfg=None)

# dataset settings
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadPanopticAnnotations',
        with_bbox=True,
        with_mask=True,
        with_seg=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[[
            dict(
                type='RandomChoiceResize',
                scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                        (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                        (736, 1333), (768, 1333), (800, 1333)],
                keep_ratio=True)
        ],
                    [
                        dict(
                            type='RandomChoiceResize',
                            scales=[(400, 1333), (500, 1333), (600, 1333)],
                            keep_ratio=True),
                        dict(
                            type='RandomCrop',
                            crop_type='absolute_range',
                            crop_size=(384, 600),
                            allow_negative_crop=True),
                        dict(
                            type='RandomChoiceResize',
                            scales=[(480, 1333), (512, 1333), (544, 1333),
                                    (576, 1333), (608, 1333), (640, 1333),
                                    (672, 1333), (704, 1333), (736, 1333),
                                    (768, 1333), (800, 1333)],
                            keep_ratio=True)
                    ]]),
    dict(type='PackDetInputs')
]

train_dataloader = dict(
    batch_size=1, num_workers=1, dataset=dict(pipeline=train_pipeline))

val_dataloader = dict(batch_size=1, num_workers=1)

test_dataloader = val_dataloader

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.0001,
        weight_decay=0.0001,
        eps=1e-8,
        betas=(0.9, 0.999)),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1, decay_mult=1.0),
            'query_embed': dict(lr_mult=1.0, decay_mult=0.0)
        },
        norm_decay_mult=0.0),
    clip_grad=dict(max_norm=0.01, norm_type=2))

max_epochs = 75

# learning rate
param_scheduler = dict(
    type='MultiStepLR',
    begin=0,
    end=max_epochs,
    by_epoch=True,
    milestones=[50],
    gamma=0.1)

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (16 GPUs) x (1 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)
